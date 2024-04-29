# cartographer使用踩坑记录

在做多机器人项目时使用cartographer建图，用于nav2导航

[我是跟着鱼香ros学的](https://fishros.com/d2lros2/#/humble/chapt10/get_started/3.%E9%85%8D%E7%BD%AEFishBot%E8%BF%9B%E8%A1%8C%E5%BB%BA%E5%9B%BE)


#### 问题一：.lua参数
[cartographer官方参数介绍](https://google-cartographer-ros.readthedocs.io/en/latest/ros_api.html)
[大佬介绍map_frame，odom_frame，published_frame，provide_odom_frame。](https://zhuanlan.zhihu.com/p/354723907)

#### 问题二：frame设置(请先看问题一的大佬介绍)
因为我整个项目是多机器人，所以在tf和node上都设置了namespace
但是问题来了‼
**我的odom使用的是gazebo的插件libgazebo_ros_p3d.so**
```
        <plugin filename="libgazebo_ros_p3d.so" name="gazebo_ros_p3d" >
            <ros>
                <namespace>/${robot_name}</namespace>
            </ros>
            <frame_name>odom</frame_name>
            <body_name>base_link</body_name>
            <update_rate>50.0</update_rate>
            <gaussian_noise>0.01</gaussian_noise>
            <!-- initialize odometry for fake localization-->
                <xyz_offset>0 0 0</xyz_offset>
                <rpy_offset>0 0 0</rpy_offset>
        </plugin>
```
我的tf起始是**namespace/base_link**
gazebo_ros_p3d发布的topic /namespace/odom 的frame_id是odom，child_frame_id是base_link **没有加前缀‼**
这将导致：**cartographer读取的odom(即/namespace/odom)识别的child_frame_id是不带前缀的**
然后就会报错
`[cartographer logger]: W0425 22:35:51.000000 377305 tf_bridge.cpp:53] "base_link" passed to lookupTransform argument source_frame does not exist.`
注意这里是**source_frame does not exist**,我理解的就是odom指向的frame不存在，当然，**因为没有前缀，base_link肯定不存在了**
我想了两个解决办法：
1. 给cartographer的node添加namespace？
没成功
2. 修改gazebo_ros_p3d发布的topic，将child_frame_id改为有namespace前缀的
不行，给它添加的namespace只是给node和topic添加前缀，而不会给tf或frame_id添加前缀
3. 单独写一个node接收gazebo_ros_p3d发布的topic，再发布一个带有自己设定的frame_id的topic
   transform_frame_id.py
```
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
```
launch.py
```
# transform odom.frame_id and odom.child_frame_id
transform_frame_id = Node(
        package='transform_frame_id',
        executable='transform_frame_id_node',
        name='transform_frame_id_node',  # Use unique node name
        namespace=robot_name,
        parameters=[{

            'frame_id':f'{robot_name}/odom',
            'child_frame_id':f'{robot_name}/base_link'
        }],
)
```

然后.lua配置
```
include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  -- slam发布的最顶层的静态坐标系
  map_frame = "map", 
  -- 机器人的基坐标tf ,我的namespace是axebot_0
  tracking_frame = "axebot_0/base_link",
  -- true，发布额外的一个中间坐标系 map -> odom_frame -> published_frame
  provide_odom_frame = true,
  published_frame = "axebot_0/base_link",
  odom_frame = "axebot_0/odom",  -- unabled
  -- false改为true，仅发布2D位资
  publish_frame_projected_to_2d = true,
  -- false改为true，使用里程计数据
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  -- 0改为1,使用一个雷达
  num_laser_scans = 1,
  -- 1改为0，不使用多波雷达
  num_multi_echo_laser_scans = 0,
  -- 10改为1，1/1=1等于不分割
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}


-- false改为true，启动2D SLAM
MAP_BUILDER.use_trajectory_builder_2d = true

-- 0改成0.10,比机器人半径小的都忽略
TRAJECTORY_BUILDER_2D.min_range = 0.10
-- 30改成3.5,限制在雷达最大扫描范围内，越小一般越精确些
TRAJECTORY_BUILDER_2D.max_range = 3.5
-- 5改成3,传感器数据超出有效范围最大值
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.
-- true改成false,不使用IMU数据，大家可以开启，然后对比下效果
TRAJECTORY_BUILDER_2D.use_imu_data = false
-- false改成true,使用实时回环检测来进行前端的扫描匹配
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true 
-- 1.0改成0.1,提高对运动的敏感度
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(1)

-- 0.55改成0.65,Fast csm的最低分数，高于此分数才进行优化。
POSE_GRAPH.constraint_builder.min_score = 0.65
--0.6改成0.7,全局定位最小分数，低于此分数则认为目前全局定位不准确
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7

-- 设置0可关闭全局SLAM
-- POSE_GRAPH.optimize_every_n_nodes = 0

return options
```
然后就成功咯