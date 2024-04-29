#!/usr/bin/env python3
 
import rclpy
from rclpy.node import Node
 
import cv2
import numpy as np
from pynput import keyboard #引入键盘监听功能库
 
from sensor_msgs.msg import Image 
from geometry_msgs.msg import TwistStamped,Twist
from cv_bridge import CvBridge
 
class ControlNode(Node):
    
    def __init__(self,name):
        super().__init__(name)
        #初始化控制消息，设置header.frame_id
        self.action = Twist()

        #创建控制消息发布接口
        self.pub_action = self.create_publisher(Twist, "/axebot_0/omnidirectional_controller/cmd_vel_unstamped", 10)
        #创建键盘事件监听器，并启动
        self.listener = keyboard.Listener(on_press=self.on_press,on_release=self.on_release)
        self.listener.start()
    
    #键盘按键按下事件处理，按下方向键时设定线速度和角速度数据并发布
    def on_press(self, key):
        #判断是否是方向键，只处理方向键事件
        if key == keyboard.Key.up or key == keyboard.Key.down or key == keyboard.Key.left or key == keyboard.Key.right:
            if key == keyboard.Key.up:      #上：向前
                self.action.linear.x = 1.0 #设置线速度
                self.action.angular.x = 0.0  #设置角速度
            elif key == keyboard.Key.down:  #下：向后
                self.action.linear.x = -1.0 #设置线速度
                self.action.angular.x = 0.0  #设置角速度
            elif key == keyboard.Key.left:  #左：左转
                self.action.linear.x = 0.0   #设置线速度
                self.action.angular.z = 1.0  #设置角速度
            elif key == keyboard.Key.right: #右：右转
                self.action.linear.x = 0.0   #设置线速度
                self.action.angular.z = -1.0 #设置角速度
            #设置消息时间数据
            # self.action.stamp = self.get_clock().now().to_msg()

            #发布消息
            self.pub_action.publish(self.action)
     #键盘按键松开事件处理，松开方向键时设定线速度和角速度为0并发布
    def on_release(self, key):
         #判断是否是方向键，只处理方向键事件
        if key == keyboard.Key.up or key == keyboard.Key.down or key == keyboard.Key.left or key == keyboard.Key.right:
            self.action.linear.x = 0.0
            self.action.angular.z = 0.0
            # self.action.header.stamp = self.get_clock().now().to_msg()
            self.pub_action.publish(self.action)
    
def main(args=None):
    rclpy.init(args=args)
    node = ControlNode(name="control_node")
    rclpy.spin(node=node)
    rclpy.shutdown()