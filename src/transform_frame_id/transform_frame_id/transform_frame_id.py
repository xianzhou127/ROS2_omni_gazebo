#!/usr/bin/env python3
# 只适用本项目,本节点不发布tf变换

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import Odometry

class Transform_frame_id(Node):
    def __init__(self,name):
        super().__init__(name)
        self.declare_parameter('frame_id', 'odom')  # 默认值
        self.declare_parameter('child_frame_id', 'base_link')  # 默认值

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.child_frame_id = self.get_parameter('child_frame_id').get_parameter_value().string_value
        self.get_logger().info(f'Received parameters: {self.frame_id} {self.child_frame_id}')

        self.group1 = MutuallyExclusiveCallbackGroup()  # 多线程

        self.target_sub = self.create_subscription(Odometry,f"odom",self.__trans_callback,10,callback_group=self.group1)
        self.trans_pub = self.create_publisher(Odometry,f"odom_trans",10)

    def __trans_callback(self,msg):
        temp = Odometry()
        temp = msg
        temp.header.frame_id = self.frame_id
        temp.child_frame_id = self.child_frame_id
        self.trans_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args) # 初始化rclpy
    node = Transform_frame_id("Transform_frame_id_node")  # 新建一个节点

    executor = MultiThreadedExecutor(3)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
