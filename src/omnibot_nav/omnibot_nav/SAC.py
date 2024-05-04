from utils import Actor, Double_Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class SAC_countinuous():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)	# 有点像double network，为了减弱bootstrapping,
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		'C:max_size=int(1e6)'
		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=self.buffer_size, dvc=self.dvc)
		# self.replay_buffer = ExperienceReplayBuffer(state_shape=self.state_dim,action_dim=self.action_dim,max_size=self.buffer_size, device=self.dvc)

		if self.adaptive_alpha:
			# Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
			self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
			# We learn log_alpha instead of alpha to ensure alpha>0
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

	def select_action(self, state, deterministic):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
			a, _ = self.actor(state, deterministic, with_logprob=False)
		return np.array(a.cpu().numpy()[0].tolist())

	def train(self,):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
			target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)	#这里作为target=使用q_critic_target而不是q_critic,前者只更新部分参数，降低bootstrapping
			target_Q = torch.min(target_Q1, target_Q2)	#Q_seta取两个评论员网络给出的价值评价的较小的那个
			# Q_soft = r(s,a) + gamma * (Q_soft(s`,a`) - alpha * ln(pi(a`))),[-lin(pi(a`))] is entropy
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		# Update Q_critic
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) # 用二者的loss和更新两个Q_critic,double Q_critic 可以降低overestimation
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
		# Freeze critic so you don't waste computational effort computing gradients for them when update actor
		for params in self.q_critic.parameters(): params.requires_grad = False

		'为什么这里的a不是expernience里的,再次使用s获得的a与exp里的a不一样'
		a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)	# deterministic=False使用重采样
		current_Q1, current_Q2 = self.q_critic(s, a)
		Q = torch.min(current_Q1, current_Q2)

		# J_pi(seta) = E[(alpha * log_pi_a - Q(s,a))]
		a_loss = (self.alpha * log_pi_a - Q).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters(): params.requires_grad = True

		#----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
		if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha>0
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		#----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return q_loss,a_loss,alpha_loss
		'为什么没有V_net'
		# R:在updated version里没有v_net，V就是E(Q)
	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep),map_location=torch.device('cpu')))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep),map_location=torch.device('cpu')))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
	
class ExperienceReplayBuffer():
    def __init__(self, max_size, state_shape, action_dim, device='cpu'):
        self.max_size = max_size
        self.state_buffer = torch.zeros((max_size, *state_shape), dtype=torch.uint8, device=device)
        self.action_buffer = torch.zeros((max_size, action_dim), dtype=torch.float, device=device)
        self.reward_buffer = torch.zeros((max_size,), dtype=torch.float, device=device)
        self.next_state_buffer = torch.zeros((max_size, *state_shape), dtype=torch.uint8, device=device)
        self.done_buffer = torch.zeros((max_size,), dtype=torch.bool, device=device)
        self.size = 0
        self.capacity = 0

    def add(self, state, action, reward, next_state, done):
        index = self.capacity % self.max_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done
        self.size = min(self.size + 1, self.max_size)
        self.capacity += 1

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = self.state_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        next_states = self.next_state_buffer[indices]
        dones = self.done_buffer[indices]
        return states, actions, rewards, next_states, dones