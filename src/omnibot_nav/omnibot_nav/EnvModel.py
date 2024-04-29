#!/usr/bin/env python3
 
import rclpy
from rclpy.node import Node
from rclpy import qos_overriding_options

from nav_msgs.msg import Odometry,OccupancyGrid
from geometry_msgs.msg import Twist,Pose,PoseStamped
from sensor_msgs.msg import LaserScan
import tf2_ros
import tf2_geometry_msgs

import math
import numpy as np
import torch
import time

class EnvModel(Node):
    def __init__(self,):
        super().__init__('envmodel')
        self.agentrobot = 'omnibot'

        self.dis      = 1.0  # 位置精度-->判断是否到达目标的距离

        self.action_space = 3
        self.state_space = 6+2

        # 储存机器人每次的目标点 
        # self.dynamic_goal_x = np.zeros(1)
        # self.dynamic_goal_y = np.zeros(2)
        self.dynamic_goal = np.zeros(2)
        # 储存机器人每次的起点
        self.dynamic_start_pos = np.zeros(2)

        # receive laser information
        self.laser_sub = self.create_subscription(LaserScan,'/axebot_0/laser/scan',self.__laser_callback,10)

        # reveive odom information
        self.odom_sub = self.create_subscription(Odometry,'/axebot_0/odom_trans',self.__odom_callback,10)

        # reveive target point information
        self.target_sub = self.create_subscription(PoseStamped,'/goal_pose',self.__targetpose_callback,10)

        # reveive target point information
        self.course_sub = self.create_subscription(PoseStamped,'/course_pose',self.__coursepose_callback,10)


        self.map_sub = self.create_subscription(OccupancyGrid,'/map',self.__map_callback,10)

        # publish velocity command to agent
        self.agent_pub = self.create_publisher(Twist,'/axebot_0/omnidirectional_controller/cmd_vel_unstamped',10)

        # controller frequency 20
        self.control_timer = self.create_timer(0.05,self.run)

        self.step_timer = self.create_timer(1,self.step)

        # reset value
        # original agent situation
        self.robot_state = np.zeros(6)                      # x,y,seta,vx,vy,w
        self.init_state = np.zeros(2)                       # x_init,y_init,
        self.target_state = np.zeros(3)                     # x_target,y_target,d_agent2target
        self.course_state = np.zeros(3)                     # x_course,y_course,d_agent2course
        self.d_target = 0
        self.d_course = 0

        self.r          = 0.0                                  # reward
        self.action        = np.zeros(3)                         # action [vx,vy,w]
        self.done       = False                                # done
        self.map = []

    def __map_callback(self,map):
        self.get_logger().info("get map")
        self.map = map.data

    def __laser_callback(self,data):
        pass

    def __odom_callback(self,data):
        self.get_logger().info("get odom")
        # get seta
        quaternion = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w

        )
        roll, pitch, yaw =  self.euler_from_quaternion(quaternion[0],quaternion[1],quaternion[2],quaternion[3])
        
        # save state
        self.robot_state[0] = data.pose.pose.position.x
        self.robot_state[1] = data.pose.pose.position.y
        self.robot_state[2] = yaw
        self.robot_state[3] = data.twist.twist.linear.x
        self.robot_state[4] = data.twist.twist.linear.y
        self.robot_state[5] = data.twist.twist.angular.z

    def __targetpose_callback(self,data):
        print("get target pose")
        self.target_state[0] = data.pose.position.x
        self.target_state[1] = data.pose.position.y
        self.d_target               = math.sqrt(
            (self.robot_state[0]-self.target_state[0])**2 + (self.robot_state[1]-self.target_state[1])**2 /
            (self.init_state[0]-self.target_state[0])**2 + (self.init_state[1]-self.target_state[1])**2
        )
        self.target_state[2] = self.d_target

    def __coursepose_callback(self,data):
        # update initial point
        if (self.course_state[0] != data.pose.position.x) and (self.course_state[1] != data.pose.position.y):
            self.init_state[0] = self.robot_state[0]
            self.init_state[1] = self.robot_state[1]

        self.course_state[0] = data.pose.position.x
        self.course_state[1] = data.pose.position.y
        self.d_course               = math.sqrt(
            (self.robot_state[0]-self.target_state[0])**2 + (self.robot_state[1]-self.target_state[1])**2 /
            (self.init_state[0]-self.target_state[0])**2 + (self.init_state[1]-self.target_state[1])**2
        )
        self.course_state[2] = self.d_course  

    # pub
    def run(self):
        cmd = Twist()
        cmd.linear.x = self.action[0]
        cmd.linear.y = self.action[1]
        cmd.angular.z = self.action[2]
        self.agent_pub.publish(cmd)

    def get_result(self):
        if self.d_target <= 0.01:
            self.done =  1
        elif self.map[int(self.robot_state[0]*100), int(self.robot_state[1]*100)]:
            self.done = 2
        else:
            self.done = 0

    def get_reward(self):
        # update done
        self.get_result()

        # arrive the target
        if self.done == 1:
            return 100
        # have a collision
        elif self.done == 2:
            return -100
        else:
            result = -(1 * self.target_state[2])
    
    def get_state(self):
        state = []

        # laser = ...
        state = list(self.robot_state) + list(self.init_state) + list (self.target_state) + list(self.course_state)

        return np.array(state)

    # def if_done(self):

    # def reset(self):
            
    # get s_,r,d,
    def step(self, action = [0.0,0.0,0.0]):
        # state = torch.zeros(self.state_space)
        # reward = 0
        # done = False

        self.action = action

        # 0.05 s/time
        time.sleep(0.05)

        reward = self.get_reward()
        state = self.get_state()
        done = self.done
        print(state)
            
        return state, reward, done, {}
    
    def euler_from_quaternion(self, x, y, z, w):
        euler = [0, 0, 0]
        Epsilon = 0.0009765625
        Threshold = 0.5 - Epsilon
        TEST = w * y - x * z
        if TEST < -Threshold or TEST > Threshold:
            if TEST > 0:
                sign = 1
            elif TEST < 0:
                sign = -1
            euler[2] = -2 * sign * math.atan2(x, w)
            euler[1] = sign * (math.pi / 2.0)
            euler[0] = 0
        else:
            euler[0] = math.atan2(2 * (y * z + w * x),
                                    w * w - x * x - y * y + z * z)
            euler[1] = math.asin(-2 * (x * z - w * y))
            euler[2] = math.atan2(2 * (x * y + w * z),
                                    w * w + x * x - y * y - z * z)

        return euler
    
def main(args=None):
    rclpy.init(args=args)
    node = EnvModel()
    # node.step([1.0,0.0,0.0])    
    rclpy.spin(node=node)
    rclpy.shutdown()

        
            

        
            
