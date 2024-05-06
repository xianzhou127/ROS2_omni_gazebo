#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from omnibot_msgs.srv import Reset,Step,Srvdone
from std_srvs.srv import Empty
from omnibot_nav.SAC import SAC_countinuous,ReplayBuffer
from omnibot_nav.utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters,GetParameters

from datetime import datetime
import os, shutil
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm   
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
import random

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIndex', type=int, default=0, help='easy_world hard_world')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=1, help='which model to load')
parser.add_argument('--is_train', type=str2bool, default=True, help='if is trun')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(10e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=500, help='Training Frequency, in stpes')
parser.add_argument('--train_freq', type=int, default=50, help='Training frequency per second')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--buffer_size', type=int, default=int(1e6), help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

EnvName = ['easy_world','hard_world']
BrifEnvName = ['Easy', 'Hard']

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))

class DRLagent(Node):
    def __init__(self):
        super().__init__('DRL_agent')
        # declare params
        self.declare_parameters(
            namespace='',
            parameters=[
                ('train_freq',opt.train_freq),
                ('world_name',EnvName[opt.EnvIndex]),
                ('is_train',opt.is_train),
            ]
        )   
        
        # client
        self.reset = self.create_client(Reset,'/env_reset')
        self.step  = self.create_client(Step,'/env_step')
        self.init  = self.create_client(Srvdone,'/env_init')
        self.get_params = self.create_client(GetParameters,'/envmodel/get_parameters')

        # rate
        self.train_rate = self.create_rate(opt.train_freq)

        # gazebo 
        self.gazebo_pause = self.create_client(Empty,'/pause_physics')
        self.gazebo_unpause = self.create_client(Empty,'/unpause_physics')

        self.env_init()
        self.get_env_params()

    def env_init(self):
        self.get_logger().info('env init start')
        while not self.init.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('env init service not available, waiting again...')
        request = Reset.Request()
        future = self.reset.call_async(request)
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    self.get_logger().info('env init success')
                    break
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("ERROR getting init service response!")

    def env_reset(self):
        while not self.reset.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('env reset service not available, waiting again...')
        request = Reset.Request()
        future = self.reset.call_async(request)
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    res = future.result()
                    return np.array(res.state),{}
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("ERROR getting reset service response!")
    
    def env_step(self,actions):
        while not self.step.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('env step service not available, waiting again...')
        request = Step.Request()
        request.actions = actions
        future = self.step.call_async(request)
        
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    res = future.result()
                    return np.array(res.state),res.reward,res.done,{}
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("ERROR getting reset service response!")
    
    def get_env_params(self):
        self.get_logger().info('getting env params')
        while not self.get_params.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('env get params service not available, waiting again...')
        request = GetParameters.Request()    
        request.names = ['state_dim','action_dim','action_space','max_episode_steps']
        future = self.get_params.call_async(request)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    print(2)
                    res = future.result()
                    # get params
                    opt.state_dim = res.values[0].integer_value
                    opt.action_dim = res.values[1].integer_value
                    opt.max_action = res.values[2].double_array_value[1]
                    opt.min_action = res.values[2].double_array_value[0]
                    opt.max_e_steps = res.values[3].integer_value
                    self.get_logger().info(f'Env:{EnvName[opt.EnvIndex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
                                        f'max_a:{opt.max_action}  min_a:{opt.min_action}  max_e_steps:{opt.max_e_steps}')
                    return
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("ERROR getting reset service response!")

    def action_sample(self):
        return np.array([random.uniform(-2,2),random.uniform(-2,2),random.uniform(-2,2)])

    def pause_sim(self):
        while not self.gazebo_pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo pause service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_pause.call_async(req)

    def unpasue_sim(self):
        while not self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo unpause service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_unpause.call_async(req)

    def train(self):
        """
        s[]
        a[0.0,0.0,0.0]: v-x,v-y,w-z
        done: 0-continue,1-win,2-default
        r: reward
        """

        # Seed Everything
        env_seed = opt.seed
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Random Seed: {}".format(opt.seed))

        # # Build SummaryWriter to record training curves
        if opt.write:
            from torch.utils.tensorboard import SummaryWriter
            timenow = str(datetime.now())[0:-10]
            timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
            writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIndex]) + timenow
            if os.path.exists(writepath): shutil.rmtree(writepath)
            writer = SummaryWriter(log_dir=writepath)

        # # Build DRL model
        if not os.path.exists('model'): os.mkdir('model')
        agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary
        if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIndex], opt.ModelIdex)
        
        total_steps = 0
        progress_bar = tqdm(total=opt.Max_train_steps)

        while total_steps < opt.Max_train_steps:
            s, info = self.env_reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
            done = 0
            e_q_loss ,e_a_loss,e_alpha_loss = 0,0,0

            while done != 2:

                if total_steps < (5*opt.max_e_steps):
                    act = self.action_sample()  # act∈[-max,max]
                    a = Action_adapter_reverse(act, opt.max_action)  # a∈[-2,2]
                else:
                    a = agent.select_action(s, deterministic=False)  # a∈[-2,2]
                    act = Action_adapter(a, opt.max_action)  # act∈[-max,max]
                s_next, r, done, info = self.env_step(list(act))  # dw: dead&win; tr: truncated  这里会阻塞
                # print(s_next[0:2])

                agent.replay_buffer.add(s, a, r, s_next, done)
                s = s_next
                total_steps += 1
                progress_bar.update(1)      #进度条更新

                '''train if it's time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    self.pause_sim()
                    for j in range(opt.update_every):
                        q_loss , a_loss, alpha_loss = agent.train()
                        e_q_loss += q_loss
                        e_a_loss += a_loss
                        e_alpha_loss += alpha_loss
                    self.unpasue_sim()

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    self.pause_sim()
                    # ep_r = evaluate_policy(eval_env, agent, turns=3)
                    # if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    if opt.write: writer.add_scalar('e_q_loss', e_q_loss, global_step=total_steps)
                    if opt.write: writer.add_scalar('e_a_loss', e_a_loss, global_step=total_steps)
                    if opt.write: writer.add_scalar('e_alpha_loss', e_alpha_loss, global_step=total_steps)
                    # print(f'EnvName:{BrifEnvName[opt.EnvIndex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')
                    progress_bar.set_postfix({
                        # 'return' : '%.3f' % (ep_r),
                        'e_q_loss' : '%.3f' % (e_q_loss),
                        'e_a_loss' : '%.3f' % (e_a_loss),
                        'e_alpha_loss' : '%.3f' % (e_alpha_loss),
                    })
                    self.unpasue_sim()

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    self.pause_sim()
                    agent.save(BrifEnvName[opt.EnvIndex], int(total_steps/1000))
                    self.unpasue_sim()

                t = time.time()
                print(int(round(t * 1000)))    #毫秒级时间戳
                time.sleep(1/opt.train_freq)
                # env.train_freq.sleep()


def main():
    rclpy.init()
    
    '''train'''
    agent = DRLagent()
    agent.train()
    try:
        rclpy.spin(agent)
    finally:
        rclpy.shutdown()
    # def train():
    #     total_steps = 0
    #     progress_bar = tqdm(total=opt.Max_train_steps)

    #     env.pause_sim()
    #     while total_steps < opt.Max_train_steps:
    #         s, info = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
    #         done = 0
    #         e_q_loss ,e_a_loss,e_alpha_loss = 0,0,0
    #         time.sleep(0.5)
    #         env.unpasue_sim()

    #         while done != 2:
    #             if total_steps < (5*opt.max_e_steps):
    #                 act = env.action_sample()  # act∈[-max,max]
    #                 a = Action_adapter_reverse(act, opt.max_action)  # a∈[-2,2]
    #             else:
    #                 a = agent.select_action(s, deterministic=False)  # a∈[-2,2]
    #                 act = Action_adapter(a, opt.max_action)  # act∈[-max,max]
    #             s_next, r, done, info = env.step(act)  # dw: dead&win; tr: truncated  这里会阻塞
    #             # print(s_next[0:2])

    #             agent.replay_buffer.add(s, a, r, s_next, done)
    #             s = s_next
    #             total_steps += 1
    #             progress_bar.update(1)      #进度条更新

    #             '''train if it's time'''
    #             # train 50 times every 50 steps rather than 1 training per step. Better!
    #             if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
    #                 env.pause_sim()
    #                 for j in range(opt.update_every):
    #                     q_loss , a_loss, alpha_loss = agent.train()
    #                     e_q_loss += q_loss
    #                     e_a_loss += a_loss
    #                     e_alpha_loss += alpha_loss
    #                 env.unpasue_sim()

    #             '''record & log'''
    #             if total_steps % opt.eval_interval == 0:
    #                 # ep_r = evaluate_policy(eval_env, agent, turns=3)
    #                 # if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
    #                 if opt.write: writer.add_scalar('e_q_loss', e_q_loss, global_step=total_steps)
    #                 if opt.write: writer.add_scalar('e_a_loss', e_a_loss, global_step=total_steps)
    #                 if opt.write: writer.add_scalar('e_alpha_loss', e_alpha_loss, global_step=total_steps)
    #                 # print(f'EnvName:{BrifEnvName[opt.EnvIndex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')
    #                 progress_bar.set_postfix({
    #                     # 'return' : '%.3f' % (ep_r),
    #                     'e_q_loss' : '%.3f' % (e_q_loss),
    #                     'e_a_loss' : '%.3f' % (e_a_loss),
    #                     'e_alpha_loss' : '%.3f' % (e_alpha_loss),
    #                 })

    #             '''save model'''
    #             if total_steps % opt.save_interval == 0:
    #                 agent.save(BrifEnvName[opt.EnvIndex], int(total_steps/1000))

    #             # t = time.time()
    #             # print(int(round(t * 1000)))    #毫秒级时间戳
    #             time.sleep(1/opt.train_freq)
    #             # env.train_freq.sleep()
    #         env.pause_sim()
    #         # eval_env.close()

    # train()
    # scheduler = BlockingScheduler()
    # scheduler.add_job(train,"interval",seconds = 1/opt.train_freq)



if __name__ == "__main__":
    main()


