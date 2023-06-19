# june_demo

## Pre-installation

- You should use Ubuntu 20.04
- Make sure you have an NVIDIA GPU
- Install [ZED SDK](https://www.stereolabs.com/developers/release)

## Installation

1. Clone `sidewalk_mobility_demo` repo
2. Inside `src`, clone `https://github.com/stereolabs/zed-ros-wrapper` with `--recurse-submodules`
3. Build with catkin

## Usage 

- Run the sensor: `roslaunch june_demo sensor.vehicle` 
- Open RViz: `rviz -d src/june_demo/rviz/june_demo.rviz`
