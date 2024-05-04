#### 启动仿真
`ros2 launch axebot_gazebo axebot.launch.py`

#### 启动slam
`ros2 launch slam_cartographer cartographer.launch.py

#### 启动规划节点
`ros2 run omnibot_nav envmodel`

#### 加载静态地图
`ros2 run nav2_map_server map_server --ros-args --param yaml_filename:=map/easy_map.yaml`
`ros2 lifecycle set /map_server configure`
`ros2 lifecycle set /map_server activate`

https://github.com/CHH3213/Note-Ubuntu_CHH3213/issues/51