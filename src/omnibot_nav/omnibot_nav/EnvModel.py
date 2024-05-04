#!/usr/bin/env python3
 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,qos_profile_system_default,qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup,ReentrantCallbackGroup

from nav_msgs.msg import Odometry,OccupancyGrid
from geometry_msgs.msg import Twist,Pose,PoseStamped
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity,SpawnEntity,GetEntityState,SetEntityState
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
import tf2_ros
import tf2_geometry_msgs


import math
import numpy as np
import torch
import time
import xacro
import os
import random
import cv2
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
from threading import Thread
# from tin_map import OptMap
# import sys
# sys.path.append('/opt/ros/humble/lib/gazebo_ros/')
# from spawn_entity import SpawnEntityNode

from launch import LaunchDescription
robot_name = "axebot_0"

np.seterr(divide='ignore',invalid='ignore')

def get_path(pkg_name,floder): 
    return os.path.join(
        get_package_share_directory(pkg_name),
        floder
    )

map_path = get_path('axebot_description','world')

def get_xml():
    xacro_file = os.path.join(
        get_package_share_directory('axebot_description'),
        'urdf',
        'start.urdf.xacro'
    )
    return xacro.process_file(xacro_file, mappings={'robot_name_rc': robot_name}).toxml()

class EnvModel(Node):
    def __init__(self,map_name='easy_world',train_freq = 100,is_train = True ,is_test_env = False):
        # create node
        rclpy.init()      
        self.envmodel = rclpy.create_node('envmodel')

        self.agentrobot = robot_name
        self.is_train = is_train
        self.group = ReentrantCallbackGroup()  # 多线程

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        self.laser_sub = Node.create_subscription(self.envmodel,LaserScan,'/axebot_0/laser/scan',self.__laser_callback,qos_profile_sensor_data, callback_group = self.group)
        self.odom_sub = Node.create_subscription(self.envmodel,Odometry,'/axebot_0/odom_trans',self.__odom_callback,QoSProfile(depth=5), callback_group = self.group)
        if self.is_train:
            self.target_sub = Node.create_subscription(self.envmodel,PoseStamped,'/goal_pose',self.__targetpose_callback,QoSProfile(depth=5), callback_group = self.group)
        if self.is_train:
            self.course_sub = Node.create_subscription(self.envmodel,PoseStamped,'/course_pose',self.__coursepose_callback,QoSProfile(depth=5), callback_group = self.group)
        self.map_sub = Node.create_subscription(self.envmodel,OccupancyGrid,'/map',self.__map_callback,QoSProfile(depth=5), callback_group = self.group)
        self.agent_pub = Node.create_publisher(self.envmodel,Twist,'/axebot_0/omnidirectional_controller/cmd_vel_unstamped',qos_profile_system_default)

        # controller frequency 20
        self.control_timer = Node.create_timer(self.envmodel,0.05,self.run)

        # train frequency
        self.train_freq = Node.create_rate(self.envmodel,train_freq)
        if is_test_env:
            self.step_timer = Node.create_timer(self.envmodel,1,self.step)

        # gazebo
        self.gazebo_get_state = Node.create_client(self.envmodel,GetEntityState,'/get_entity_state')
        self.gazebo_set_state = Node.create_client(self.envmodel,SetEntityState,'/set_entity_state')
        self.gazebo_spawn_entity = Node.create_client(self.envmodel,SpawnEntity, '/spawn_entity')
        self.gazebo_delete_entity = Node.create_client(self.envmodel,DeleteEntity, '/delete_entity')
        self.gazebo_reset_simulation = Node.create_client(self.envmodel,Empty, '/reset_simulation')
        self.gazebo_reset_world = Node.create_client(self.envmodel,Empty, '/reset_world')
        self.gazebo_pause = Node.create_client(self.envmodel,Empty,'/pause_physics')
        self.gazebo_unpause = Node.create_client(self.envmodel,Empty,'/unpause_physics')
        self.robot_xml = get_xml()

        """************************************************************
        ** Initialise state
        ************************************************************"""
        # map
        # cvmap = cv2.imread(map_path + '/' + map_name+".pgm", cv2.IMREAD_GRAYSCALE)
        # self.map = OptMap(cvmap)
        # self.map.create_triangle_map(100)
        # self.map_size = len(self.map.triangle_map)
        self.map_size = 600

        # reset value
        # original agent situation
        self.robot_state = np.zeros(6)                      # x,y,seta,vx,vy,w
        self.init_state = np.zeros(6)                       # x_init,y_init,seta_init,vx_init,vy_init,w_init
        self.init_state_cut = np.zeros(6)                   
        self.target_state = np.zeros(4)                     # x_target,y_target,seta_target,d_agent2target
        self.target_state_cut = np.zeros(4)                 # current rarget state
        self.course_state = np.zeros(3)                     # x_course,y_course,d_agent2course
        self.course_state_cut = np.zeros(3)
        self.laser_state = np.ones(361)                    # laser sample,obstacle_distance
        self.map_state = np.zeros(self.map_size)
        self.d_target = 0
        self.d_course = 0
        self.obstacle_distance = 1

        self.r          = 0.0                                  # reward
        self.action        = np.zeros(3)                         # action [vx,vy,w]
        self.done       = 0                                # done

        self.dis      = 1.0  # 位置精度-->判断是否到达目标的距离
        

        self.action_dim = 3
        self.action_space=[-2,2]
        self.state_dim = 6+6+4+3+361+self.map_size
        self.d_target_idx = 6+6+4-1
        self.d_course_idx = 6+6+4+3-1
        self.d_obstacle_idx = 6+6+4+3+361-1
        self.r_seta_idx = 3-1
        self.t_seta_idx = 6+6+3-1
        self._max_episode_steps = 1000

        self.init_env()

        # if is_test_env:
        #     rclpy.spin(self.envmodel)
        # else:
            # executor = MultiThreadedExecutor(3)
            # executor.add_node(self.envmodel)



    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def init_env(self):
        self.reset()
        # self.get_init_state()
        if self.is_train:
            self.set_target()

    def __map_callback(self,map):
        self.envmodel.get_logger().info("get map")
        self.map = map.data

    def __laser_callback(self,data):
        self.obstacle_distance = 1
        for i in range(360):
                self.laser_state[i] = np.clip(float(data.ranges[i]) / 10.0, 0, 1)
                if self.laser_state[i] < self.obstacle_distance:
                    self.obstacle_distance = self.laser_state[i]
        self.obstacle_distance *= 10.0
        self.laser_state[360] = self.obstacle_distance
        # self.envmodel.get_logger().info(f"obstacle distance{self.obstacle_distance}")

    def __odom_callback(self,data):
        # self.envmodel.get_logger().info("get odom")
        # get seta
        roll, pitch, yaw =  quaternion2euler(data.pose.pose)
        
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
        roll, pitch, yaw =  quaternion2euler(data.pose)
        self.target_state[2] = yaw

    def __coursepose_callback(self,data):
        self.course_state[0] = data.pose.position.x
        self.course_state[1] = data.pose.position.y
        
    # pub velocity command
    def run(self):
        cmd = Twist()
        cmd.linear.x = self.action[0]
        cmd.linear.y = self.action[1]
        cmd.angular.z = self.action[2]
        self.agent_pub.publish(cmd)

    def set_target(self):
        self.target_state[0] = random.uniform(5.5,8.0)    # x
        self.target_state[1] = 18.0                       # y
        self.target_state[2] = random.uniform(0.0,2*math.pi)   # seta
        # self.envmodel.get_logger().info(f'target point:{self.target_state}')

    def get_init_state(self):
        self.init_state = self.robot_state
        # self.envmodel.get_logger().info(f'target point:{self.target_state}')

    def get_target_state(self):
        self.target_state_cut = self.target_state
        self.d_target               = math.sqrt(
            ((self.robot_state[0]-self.target_state_cut[0])**2 + (self.robot_state[1]-self.target_state_cut[1])**2) /
            ((self.init_state[0]-self.target_state_cut[0])**2 + (self.init_state[1]-self.target_state_cut[1])**2)
        )
        self.target_state_cut[3] = self.d_target

    def get_course_state(self):
        # update initial point
        if not np.array_equal(self.course_state, self.course_state_cut):
            self.init_state_cut = self.robot_state

        self.course_state_cut = self.course_state
        # self.d_course               = math.sqrt(
        #     ((self.robot_state[0]-self.course_state_cut[0])**2 + (self.robot_state[1]-self.course_state_cut[1])**2) /
        #     ((self.init_state_cut[0]-self.course_state_cut[0])**2 + (self.init_state_cut[1]-self.course_state_cut[1])**2)
        # )
        # self.course_state_cut[2] = self.d_course  
        self.course_state_cut[2] = 0

    def get_result(self,state):
        done = 0
        # win distance_error < 1cm degree_error < 1 degree
        if state[self.d_target_idx] <= 0.01 and abs(state[self.r_seta_idx] - state[self.t_seta_idx])/math.pi*180 <= 1:
            done =  1
        # lose
        elif state[self.d_obstacle_idx] < 0.14: #obstacle_distance < 1cm ,
            done = 2
        # elif self.map.is_in_ob_by_binary(state[0:2]):
        #     done = 2
        else:
            done = 0
        return done

    def get_reward(self,state,done):
        target_dis = state[self.d_target_idx]         # [0,1]
        target_yaw = abs(state[self.r_seta_idx] - state[self.t_seta_idx])/math.pi*180         # [0,360]
        angular_v = state[6-1]
        course_dis = state[self.d_course_idx]
        obstacle_dis = state[self.d_obstacle_idx] # [0.12,10]

        # [-1,0] distance to target
        r_d_target = -1 * target_dis       

        # [-0.72,0] 
        r_target_yaw = -0.002 * target_yaw     

        # [-0.05,-0.025] distance to obstacle
        if obstacle_dis >= 0.20 and obstacle_dis < 0.4:
            r_obstacle = -0.01 * 1/obstacle_dis
        # [-0.67,-0.5] 
        elif obstacle_dis >= 0.15 and obstacle_dis < 0.20:
            r_obstacle = -0.1 * 1/obstacle_dis
        else:
            r_obstacle = 0

        # [-0.02,0] lower angular velocity
        r_v_angular = -0.01 * (abs(angular_v))

        reward = r_d_target+ r_target_yaw + r_obstacle + r_v_angular
        # self.envmodel.get_logger().info(f'{r_d_target}/{r_target_yaw}/{r_obstacle}/{r_v_angular}')
        # arrive the target
        if done == 1:
            reward += 10000
        # have a collision
        elif done == 2:
            reward += -10000

        return reward
    
    def get_state(self):
        # run node one time and sub info
        

        state = []
        # self.get_init_state()
        # self.get_robot_state() robot_callback
        self.get_target_state()
        self.get_course_state()
        # self.get_laser_state() laser_callback
        # self.get_map_state() 

        state = list(self.robot_state) + list(self.init_state) + list (self.target_state) + list(self.course_state) + \
                list(self.laser_state) + list(self.map_state)

        return np.array(state)

    # def if_done(self):

    def delete_entity(self):
        self.envmodel.get_logger().info('delete entity')
        while not self.gazebo_delete_entity.wait_for_service(timeout_sec=1.0):
            self.envmodel.get_logger().info('gazebo delete entity service not available, waiting again...')
        objstate = DeleteEntity.Request()
        objstate.name = self.agentrobot
        self.gazebo_delete_entity.call_async(objstate)

    def spawn_entity(self):
        self.envmodel.get_logger().info('spawn entity')
        while not self.gazebo_spawn_entity.wait_for_service(timeout_sec=1.0):
            self.envmodel.get_logger().info('gazebo spwan entity service not available, waiting again...')
        objstate = SpawnEntity.Request()
        objstate.name = self.agentrobot
        objstate.xml  = self.robot_xml
        objstate.robot_namespace = self.agentrobot

        pose = Pose()
        pose.orientation.w = 1.0
        objstate.initial_pose = pose
        objstate.reference_frame = "world"
        try:
            self.gazebo_spawn_entity.call_async(objstate)
        except Exception as e:
            print('spawn model failed') 

    def reset_simulation(self):
        while not self.gazebo_reset_simulation.wait_for_service(timeout_sec=1.0):
            self.envmodel.get_logger().info('gazebo_reset_simulation service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_reset_simulation.call_async(req)
    
    def reset_world(self):
        while not self.gazebo_reset_world.wait_for_service(timeout_sec=1.0):
            self.envmodel.get_logger().info('gazebo_reset_world service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_reset_world.call_async(req)

    def pause_sim(self):
        while not self.gazebo_pause.wait_for_service(timeout_sec=1.0):
            self.envmodel.get_logger().info('gazebo pause service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_pause.call_async(req)

    def unpasue_sim(self):
        while not self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
            self.envmodel.get_logger().info('gazebo unpause service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_unpause.call_async(req)


    def action_sample(self):
        return np.array([random.uniform(-2,2),random.uniform(-2,2),random.uniform(-2,2)])

    # reset model
    def reset(self):
        self.envmodel.get_logger().info('reset env')
        while not self.gazebo_set_state.wait_for_service(timeout_sec=1.0):
            self.envmodel.get_logger().info('gazebo set state service not available, waiting again...')
        objstate = SetEntityState.Request()
        objstate.state.name = self.agentrobot
        objstate.state.pose.position.x = random.uniform(1.0,8.0)
        objstate.state.pose.position.y = 0.0
        objstate.state.pose.position.z = 0.0
        quaternion = euler2quaternion([0.0,0.0,random.uniform(0.0,2*math.pi)])
        objstate.state.pose.orientation.x = quaternion[0]
        objstate.state.pose.orientation.y = quaternion[1]
        objstate.state.pose.orientation.z = quaternion[2]
        objstate.state.pose.orientation.w = quaternion[3]
        objstate.state.twist.linear.x = 0.0
        objstate.state.twist.linear.y = 0.0
        objstate.state.twist.linear.z = 0.0
        objstate.state.twist.angular.x = 0.0
        objstate.state.twist.angular.y = 0.0
        objstate.state.twist.angular.z = 0.0
        objstate.state.reference_frame = ""

        self.gazebo_set_state.call_async(objstate)
        # clean stack
        for i in range(10):
            rclpy.spin_once(self.envmodel)
        state = self.get_state()

        return state,{}

    # get s_,r,d,
    def step(self, action = np.array([0.0,0.0,0.0])):
        # 1 step
        self.action = action
        self.run()
        while rclpy.ok():
            rclpy.spin_once(self.envmodel)

            state = self.get_state()
            done = self.get_result(state)
            reward = self.get_reward(state,done)
            # self.envmodel.get_logger().info(f'robot state: {state[0:6]} \n init state: {state[6:6+6]} \n target state: {state[12:12+4]} \n obstacle distance {state[380-1]}')
            self.envmodel.get_logger().info(f'done: {done} reward: {reward} o_d: {state[380-1]}')
            # self.envmodel.get_logger().info(f'{state[376:379]}')
            
            return state, reward, done, {}
    
    def close(self):
        # self.executor.shutdown()
        self.envmodel.destroy_node()
        rclpy.shutdown()

def quaternion2euler(pose:Pose):
    quaternion = [
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z,
    pose.orientation.w    
    ]
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=False)
    return euler
    
def euler2quaternion(euler):
    # degree measure
    r = R.from_euler('xyz', euler, degrees=False)
    quaternion = r.as_quat()
    return quaternion


# def euler_from_quaternion(pose:Pose):
#     euler = [0, 0, 0]
#     Epsilon = 0.0009765625
#     Threshold = 0.5 - Epsilon
#     x = pose.orientation.x
#     y = pose.orientation.y
#     z = pose.orientation.z
#     w = pose.orientation.w     

#     TEST = w * y - x * z
#     if TEST < -Threshold or TEST > Threshold:
#         if TEST > 0:
#             sign = 1
#         elif TEST < 0:
#             sign = -1
#         euler[2] = -2 * sign * math.atan2(x, w)
#         euler[1] = sign * (math.pi / 2.0)
#         euler[0] = 0
#     else:
#         euler[0] = math.atan2(2 * (y * z + w * x),
#                                 w * w - x * x - y * y + z * z)
#         euler[1] = math.asin(-2 * (x * z - w * y))
#         euler[2] = math.atan2(2 * (x * y + w * z),
#                                 w * w + x * x - y * y - z * z)

#     return euler
    
def main(args=None):
    node = EnvModel(is_test_env=True)
    

# if __name__ == "__main__":
#     main()

        
            

        
            
