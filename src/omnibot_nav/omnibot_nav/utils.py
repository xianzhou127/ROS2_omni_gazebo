import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import cv2

def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.flatten()
    state = state.astype(float)
    state /= 255.0
    return state

def build_net(layer_shape, hidden_activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

def build_Covnet_a(output_dim,hidden_activation,output_activation):
	layers = nn.Sequential(
		nn.Conv2d(3,6,kernel_size=(7, 7), stride=3), hidden_activation(),
		nn.MaxPool2d(kernel_size=(2, 2)),
		nn.Conv2d(6,12,kernel_size=(4, 4)), hidden_activation(),
		nn.MaxPool2d(kernel_size=(2, 2)),
		nn.Flatten(),	#6*6*12
		nn.Linear(432,216), output_activation(),
		# nn.Linear(216,output_dim),
	)
	# for j in range(len(layer_shape)-1):
	# 	act = hidden_activation if j < len(layer_shape)-2 else output_activation
	# 	layers += [nn.Conv2d(layer_shape[j], layer_shape[j+1],), act()]
	# return nn.Sequential(*layers)
	return layers

def build_Covnet_q(input_channels, action_dim,output_dim,hidden_activation,output_activation):
	layers = nn.Sequential(
		nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),hidden_activation(),
		nn.Conv2d(16, 32, kernel_size=4, stride=2), hidden_activation(),
		nn.Linear(32 * 6 * 6 + action_dim, 256), output_activation(), # 将动作维度添加到全连接层输入中，输入不是2d，没办法加到cov2d里
		nn.Linear(256, output_dim),
	)
	return layers

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
		super(Actor, self).__init__()
		# (980,256,256)
		layers = [state_dim] + list(hid_shape)

		self.a_net = build_net(layers, hidden_activation, output_activation)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		# self.a_net = build_Covnet_a(hid_shape[0], hidden_activation, output_activation)
		# self.mu_layer = nn.Linear(hid_shape[0], action_dim)
		# self.log_std_layer = nn.Linear(hid_shape[0], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def forward(self, state, deterministic, with_logprob):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习，简单粗暴的去除极值
		# we learn log_std rather than std, so that exp(log_std) is always > 0
		std = torch.exp(log_std)
		dist = Normal(mu, std)
		if deterministic: u = mu
		else: u = dist.rsample()

		'''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
		a = torch.tanh(u)
		if with_logprob:
			# Get probability density of logp_pi_a from probability density of u:
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			# softplus(-2*u)=ln(1+e^(-2*u)),np.log(2)=ln(2),softplus 函数的优点之一是它在 xx 接近负无穷大时，函数的导数接近于 0，这有助于减缓梯度的消失问题
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)	#不变
		else:
			logp_pi_a = None

		return a, logp_pi_a

class Double_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

		# self.Q_1 = build_Covnet_q(state_dim[0],action_dim,1, nn.ReLU, nn.Identity)
		# self.Q_2 = build_Covnet_q(state_dim[0],action_dim,1, nn.ReLU, nn.Identity)

	def forward(self, state, action):
		  
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		  
		# q1 = self.Q_1(state,action.unsqueeze(0))
		# q2 = self.Q_2(state,action.unsqueeze(0))
		return q1, q2

#reward engineering for better training
def Reward_adapter(r, EnvIdex):
	# For Pendulum-v0
	if EnvIdex == 0:
		r = (r + 8) / 8

	# For LunarLander
	elif EnvIdex == 1:
		if r <= -100: r = -10

	# For BipedalWalker
	elif EnvIdex == 4 or EnvIdex == 5:
		if r <= -100: r = -1
	return r


def Action_adapter(a,max_action):
	#from [-1,1] to [-max,max]
	return  a*max_action

def Action_adapter_reverse(act,max_action):
	#from [-max,max] to [-1,1]
	return  act/max_action


def evaluate_policy(env, agent, turns = 3):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		'C'
		s = process_state_image(s)
		done = False
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			s_next, r, dw, tr, info = env.step(a)
			'C'
			s_next = process_state_image(s_next)
			done = (dw or tr)

			total_scores += r
			s = s_next
	return int(total_scores/turns)


def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')