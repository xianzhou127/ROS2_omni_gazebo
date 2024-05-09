#!/usr/bin/env python3
 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,qos_profile_system_default,qos_profile_sensor_data,qos_profile_services_default
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup,ReentrantCallbackGroup

from rcl_interfaces.srv import GetParameters
from nav_msgs.msg import Odometry,OccupancyGrid
from geometry_msgs.msg import Twist,Pose,PoseStamped
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity,SpawnEntity,GetEntityState,SetEntityState
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
from omnibot_msgs.srv import State,Step,Srvdone

import tf2_ros
import tf2_geometry_msgs


import math
import numpy as np
import time
import xacro
import os
import random
import cv2
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
from threading import Thread
from omnibot_nav.tin_map import OptMap
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
    def __init__(self,is_test_env = False):
        # create node
        super().__init__("envmodel")

        self.agentrobot = robot_name
        self.is_train = True    # 
        self.train_freq = 1000  # unused

        self.declare_parameter('map_name','easy_world')
        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        self.laser_sub = self.create_subscription(LaserScan,'/axebot_0/laser/scan',self.__laser_callback,qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry,'/axebot_0/odom_trans',self.__odom_callback,QoSProfile(depth=5))
        self.target_sub = self.create_subscription(PoseStamped,'/goal_pose',self.__targetpose_callback,QoSProfile(depth=5))
        self.course_sub = self.create_subscription(PoseStamped,'/course_pose',self.__coursepose_callback,QoSProfile(depth=5))
        # self.map_sub = self.create_subscription(OccupancyGrid,'/map',self.__map_callback,QoSProfile(depth=5))

        self.target_pub = self.create_publisher(PoseStamped,'/target_pose',QoSProfile(depth=5))
        # train frequency
        self.train_rate = self.create_rate(1000)
        # if is_test_env:
        #     self.step_timer = self.create_timer(1,self.step)

        # gazebo
        self.gazebo_get_state = self.create_client(GetEntityState,'/get_entity_state')
        self.gazebo_set_state = self.create_client(SetEntityState,'/set_entity_state')
        self.gazebo_spawn_entity = self.create_client(SpawnEntity, '/spawn_entity')
        self.gazebo_delete_entity = self.create_client(DeleteEntity, '/delete_entity')
        self.gazebo_reset_simulation = self.create_client(Empty, '/reset_simulation')
        self.gazebo_reset_world = self.create_client(Empty, '/reset_world')
        self.gazebo_pause = self.create_client(Empty,'/pause_physics')
        self.gazebo_unpause = self.create_client(Empty,'/unpause_physics')
        self.robot_xml = get_xml()

        # service
        self.env_reset = self.create_service(Empty,'/env_reset',self.reset,qos_profile=qos_profile_services_default)
        self.env_step = self.create_service(Step,'/env_step',self.step,qos_profile=qos_profile_services_default)
        self.env_init = self.create_service(Empty,'/env_init',self.init,qos_profile=qos_profile_services_default)
        self.env_state = self.create_service(State,'/env_state',self.get_state,qos_profile=qos_profile_services_default)

        # cline
        self.get_params = self.create_client(GetParameters,'/DRL_agent/get_parameters')

        # params
        self.map_name = self.get_parameter('map_name').get_parameter_value().string_value
        print(self.map_name)
        """************************************************************
        ** Initialise state
        ************************************************************"""
        # map
        cvmap = cv2.imread(map_path + '/' + self.map_name +".pgm", cv2.IMREAD_GRAYSCALE)
        # print(cvmap.shape)
        self.map = OptMap(cvmap)
        self.map.create_triangle_map(100)
        self.map_state = np.array(self.map.triangle_map).reshape(-1)
        self.map_size = len(self.map_state)
        # print(self.map.triangle_map)
        # self.map_size = 600

        # reset value
        # original agent situation
        self.robot_state = np.zeros(6)                      # x,y,seta,vx,vy,w
        self.init_state = np.zeros(6)                       # x_init,y_init,seta_init,vx_init,vy_init,w_init
        self.init_state_cut = np.zeros(6)                   
        self.target_state = np.zeros(4)                     # x_target,y_target,seta_target,d_agent2target
        self.target_state_cut = np.zeros(4)                 # current rarget state
        self.course_state = np.zeros(3)                     # x_course,y_course,d_agent2course
        self.course_state_cut = np.zeros(3)
        self.laser_state = np.zeros(361)                    # laser sample,obstacle_distance
        self.d_target = 0
        self.d_course = 0
        self.obstacle_distance = 1

        

        self.r          = 0.0                                  # reward
        self.action        = np.zeros(3)                         # action [vx,vy,w]
        self.done       = False                                # done
        self.success       = False                                # success

        self.dis      = 1.0  # 位置精度-->判断是否到达目标的距离
        

        self.action_dim = 3
        self.action_space=[-2.0,2.0]
        self.state_dim = 6+6+4+3+361+self.map_size+6    # + action[3] pre_action[3]
        self.d_target_idx = 6+6+4-1
        self.d_course_idx = 6+6+4+3-1
        self.d_obstacle_idx = 6+6+4+3+361-1
        self.r_seta_idx = 3-1
        self.t_seta_idx = 6+6+3-1
        self._max_episode_steps = 1000

        # param
        self.declare_parameters(
            namespace='',
            parameters= [
                ('state_dim', self.state_dim),
                ('action_dim', self.action_dim),
                ('max_episode_steps', self._max_episode_steps),
                ('action_space', self.action_space)
            ]
        )


    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def init(self,rquest,response):
        # self.get_parameters()

        # update train rate
        # self.train_rate.destroy()
        # self.train_rate = self.create_rate(self.train_freq)

        if self.is_train:
            self.set_target()

        # self.get_init_state()
        return response

    def get_parameters(self):
        while not self.get_params.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('env get params service not available, waiting again...')
        request = GetParameters.Request()    
        request.names = ['is_train','train_freq']
        future = self.get_params.call_async(request)
        
        time.sleep(0.01)
        print(0)
        if future.done():
            if future.result() is not None:
                res = future.result()
                # get params
                self.is_train = res.values[0].bool_value
                self.train_freq = res.values[1].integer_value
                self.get_logger().info(f'is_train:{self.is_train}  train_freq:{self.train_freq}')
                return
            else:
                self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting reset service response!")

    def __map_callback(self,map):
        self.get_logger().info("get map")
        self.map = map.data

    def __laser_callback(self,data):
        self.obstacle_distance = 1
        for i in range(360):
                self.laser_state[i] = np.clip(float(data.ranges[i]) / 10.0, 0, 1)
                if self.laser_state[i] < self.obstacle_distance:
                    self.obstacle_distance = self.laser_state[i]
        self.obstacle_distance *= 10.0
        self.laser_state[360] = self.obstacle_distance
        # self.get_logger().info(f"obstacle distance{self.obstacle_distance}")

    def __odom_callback(self,data):
        # self.get_logger().info("get odom")
        # get seta
        roll, pitch, yaw =  quaternion2euler(data.pose.pose)
        
        # save state
        self.robot_state[0] = data.pose.pose.position.x
        self.robot_state[1] = data.pose.pose.position.y
        self.robot_state[2] = yaw
        self.robot_state[3] = data.twist.twist.linear.x
        self.robot_state[4] = data.twist.twist.linear.y
        self.robot_state[5] = data.twist.twist.angular.z
        # self.get_logger().info(f"odom update {self.robot_state[0]}")

    def __targetpose_callback(self,data):
        print("get target pose")
        self.target_state[0] = data.pose.position.x
        self.target_state[1] = data.pose.position.y
        roll, pitch, yaw =  quaternion2euler(data.pose)
        self.target_state[2] = yaw

    def __coursepose_callback(self,data):
        self.course_state[0] = data.pose.position.x
        self.course_state[1] = data.pose.position.y

    def set_target(self):
        if self.map_name == "hard_world":
            self.target_state[0] = random.uniform(5.5,8.0)    # x
            self.target_state[1] = random.uniform(17.9,18.1)                       # y
            self.target_state[2] = random.uniform(0.0,2*math.pi)   # seta
        elif self.map_name == "easy_world":
            self.target_state[0] = random.uniform(0.0,2.7)    # x
            self.target_state[1] = random.uniform(4.5,4.7)                       # y
            self.target_state[2] = random.uniform(0.0,2*math.pi)   # seta
        # self.get_logger().info(f'target point:{self.target_state}')
        self.get_logger().info(f'target pose {self.target_state}')
        target_pose = PoseStamped()
        target_pose.pose.position.x = self.target_state[0] 
        target_pose.pose.position.y = self.target_state[1] 
        target_pose.pose.orientation.z = self.target_state[2]
        target_pose.header.frame_id = 'map' 
        self.target_pub.publish(target_pose)

    def get_init_state(self):
        self.init_state = self.robot_state
        # self.get_logger().info(f'target point:{self.target_state}')

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
        done = False
        success = False
        # win distance_error < 1cm degree_error < 1 degree
        if state[self.d_target_idx] <= 0.01 and abs(state[self.r_seta_idx] - state[self.t_seta_idx])/math.pi*180 <= 1:
            done = True
            success = True
        # lose
        elif state[self.d_obstacle_idx] < 0.14: #obstacle_distance < 1cm ,
            done = True
            success = False
        # elif self.map.is_in_ob_by_binary(state[0:2]):
        #     done = 2

        return done,success

    def get_reward(self,state,done,success):
        
        target_dis = state[self.d_target_idx]         # [0,1]
        target_yaw = abs(state[self.r_seta_idx] - state[self.t_seta_idx])/math.pi*180         # [0,360]
        angular_v = state[6-1]
        course_dis = state[self.d_course_idx]
        obstacle_dis = state[self.d_obstacle_idx] # [0.12,10]

        # [-2,0] distance to target
        r_d_target = -2 * target_dis       

        # [-0.9,0] 
        if target_dis < 0.2:
            r_target_yaw = -0.001 * target_yaw     
        else:
            r_target_yaw = 0

        # [-0.005,-0.0025] distance to obstacle
        if obstacle_dis >= 0.20 and obstacle_dis < 0.4:
            r_obstacle = -0.001 * 1/obstacle_dis
        # [-0.067,-0.05] 
        elif obstacle_dis >= 0.14 and obstacle_dis < 0.20:
            r_obstacle = -0.1 * 1/obstacle_dis
        else:
            r_obstacle = 0

        # [-0.02,0] lower angular velocity
        # r_v_angular = -0.01 * (abs(angular_v))
        r_v_angular = 0

        reward = r_d_target+ r_target_yaw + r_obstacle + r_v_angular
        self.get_logger().info(f'{r_d_target}/{r_target_yaw}/{r_obstacle}/{r_v_angular}')
        # arrive the target
        if done == True and success == True:
            reward += 10000
        # have a collision
        elif done == True and success == False:
            reward += -10000

        return reward
    
    def get_state(self,request,response):
        # self.get_init_state()
        # self.get_robot_state() robot_callback
        self.get_target_state()
        self.get_course_state()
        # self.get_laser_state() laser_callback
        # self.get_map_state() 
        response.state = list(self.robot_state) + list(self.init_state_cut) + list (self.target_state_cut) + list(self.course_state_cut) + \
                list(self.laser_state) + list(self.map_state) + list(request.action) + list(request.pre_action)

        return response

    # def if_done(self):

    def delete_entity(self):
        self.get_logger().info('delete entity')
        while not self.gazebo_delete_entity.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo delete entity service not available, waiting again...')
        objstate = DeleteEntity.Request()
        objstate.name = self.agentrobot
        self.gazebo_delete_entity.call_async(objstate)

    def spawn_entity(self):
        self.get_logger().info('spawn entity')
        while not self.gazebo_spawn_entity.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo spwan entity service not available, waiting again...')
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
            self.get_logger().info('gazebo_reset_simulation service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_reset_simulation.call_async(req)
    
    def reset_world(self):
        while not self.gazebo_reset_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo_reset_world service not available, waiting again...')
        req = Empty.Request()
        self.gazebo_reset_world.call_async(req)

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
    # reset model
    def reset(self,request,response):
        self.pause_sim()
        self.get_logger().info('reset env')
        while not self.gazebo_set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo set state service not available, waiting again...')
        objstate = SetEntityState.Request()
        objstate.state.name = self.agentrobot
        if self.map_name == "easy_world":
            objstate.state.pose.position.x = random.uniform(0.0,2.7)
        elif self.map_name == "hard_world":
            objstate.state.pose.position.x = random.uniform(1.0,8.0)
        objstate.state.pose.position.y = 0.0
        objstate.state.pose.position.z = 0.0
        yaw = random.uniform(0.0,2*math.pi)
        quaternion = euler2quaternion([0.0,0.0,yaw])
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
        time.sleep(0.1)
        # reset init state
        self.init_state[0] = objstate.state.pose.position.x
        self.init_state[2] = yaw
        self.init_state_cut = self.init_state

        # reset target state
        self.set_target()

        self.unpasue_sim()
        self.get_logger().info('reset finish')
        return response

    # get s_,r,d,
    def step(self, request, response):
        # 1 step
        # self.train_rate.sleep()
        action = State.Request()
        action.action = request.action
        action.pre_action = request.pre_action
        state = State.Response()
        response.state = self.get_state(action,state).state
        response.done,response.success = self.get_result(response.state)
        response.reward = self.get_reward(response.state,response.done,response.success)
        self.get_logger().info(f'{response.done} {response.success}')
        # self.get_logger().info(f'robot state: {state[0:6]} \n init state: {state[6:6+6]} \n target state: {state[12:12+4]} \n obstacle distance {state[380-1]}')
        # self.get_logger().info(f'done: {response.done} reward: {response.reward} o_d: {response.state[380-1]}')
        # self.get_logger().info(f'{state[376:379]}')
        
        return response
    
    def close(self):
        # self.executor.shutdown()
        self.destroy_node()
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
    rclpy.init(args=args)
    node = EnvModel(is_test_env=True)
    rclpy.spin(node)
    # executor = MultiThreadedExecutor(1)
    # executor.add_node(node)
    # executor.spin()
    node.destroy_node()
    rclpy.shutdown()

# if __name__ == "__main__":
#     main()

        
            

        
            
