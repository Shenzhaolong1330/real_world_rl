#!/bin/bash

# Kill any existing processes
echo "Stopping any existing processes..."
pkill -f "franka_server" 2>/dev/null
sleep 1

cd /root/hil-serl/serl_robot_infra/robot_servers

# source the catkin_ws that contains the serl_franka_controllers package
source /opt/venv/franka-0.15.0/franka_catkin_ws/devel/setup.bash

# Set ROS master URI to localhost
export ROS_MASTER_URI=http://localhost:11311

# Check if roscore is running
if ! pgrep -x "rosmaster" > /dev/null; then
    echo "Starting roscore..."
    roscore &
    sleep 2
else
    echo "roscore already running"
fi

# Check serial port
if [ ! -e /dev/ttyUSB0 ]; then
    echo "Warning: /dev/ttyUSB0 not found!"
    echo "Please check the RS-485 to USB connection"
fi

# script to start http server and ros controller
echo "Starting franka_server with Robotiq gripper (RTU mode)..."
python franka_server.py \
    --gripper_type="Robotiq" \
    --gripper_use_rtu=true \
    --gripper_device="/dev/ttyUSB0" \
    --robot_ip="172.16.0.2" \
    --flask_url="0.0.0.0" \
    --ros_port="11311"