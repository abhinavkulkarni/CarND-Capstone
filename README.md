This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

## Team
Our team name is **Vulcan/Braking Bad**.

#### Team members
|Name            |Udacity account email|
|----------------- |-----------------------------|
|Abhinav Kulkarni | abhinavkulkarni[AT]gmail.com |
|Andrej Georgievski | andrej.zgeorgievski[AT]gmail.com |
|Kamal Gupta | kamalgupta308[AT]gmail.com |
|Praveen Gunasekaran | praveenraj49[AT]gmail.com |
|Santosh Kumar | sanchelseaster[AT]gmail.com |

## Implementation
The goal of the project is to enable a car to drive on its own. A typical self-driving car has following components:
* Sensors - are the hardware components the car uses to observe the world
* Perception module - processes data from one or more sensors and gives out structured information that can further be used in path planning or control
* Planning module - is responsible for both high and low level decisions about what actions should car perform
* Control - ensures that car follows the path set by planning module even with latency between various commands.

![Alt text](imgs/carla_architecture.png?raw=true "Architecture of Carla")

The scope of this project is limited to driving Carla around a set of way-points on the road while stopping appropriately at the traffic lights. We will be using ROS (Robot Operating System) to implement various components of this problem.


#### DBW Node

![Alt text](imgs/dbw-node-ros-graph.png?raw=true "DBW Node")

#### Waypoint Updater Node

![Alt text](imgs/waypoint-updater-ros-graph.png?raw=true "Waypoint Updater Node")

#### Traffic Light Detection Node

![Alt text](imgs/tl-detector-ros-graph.png?raw=true "Traffic Light Detection Node")


Here is the final state architecture diagram of various topics and nodes we have used in ROS

![Alt text](imgs/final-project-ros-graph-v2.png?raw=true "Architecture of Carla")


## Installation
Please use **one** of the two installation options, either native **or** docker installation.

#### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

#### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

#### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

## Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
(Only for Workspace) Install `ros-kinetic-dbw` package
```
sudo apt-get update
sudo apt-get install -y ros-kinetic-dbw-mkz-msgs
```

3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

## Testing components

#### Waypoint Updater Node Partial
Run `roslaunch  launch/styx.launch` in one of the terminals, On second terminal,
* either run `rostopic echo /final_waypoints`
* run the simulator and see bunch of green dots appearing in front of the car

#### DBW Node
1. Download the rosbag file from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/reference.bag.zip) and save it as `/home/workspace/CarND-Capstone/data/dbw_test.rosbag.bag`
2. Go to the directory `/home/workspace/CarND-Capstone/ros`
3. Run `roslaunch src/twist_controller/launch/dbw_test.launch`
4. If ran correctly, this generates 3 files in the directory `/home/workspace/CarND-Capstone/ros/src/twist_controller`
   * `brakes.csv`
   * `throttles.csv`
   * `steers.csv`
   
#### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

## Debugging
 
While compiling on Udacity workspace

if you get following error message
```
CMake Warning at /opt/ros/kinetic/share/catkin/cmake/catkinConfig.cmake:76 (find_package):
  Could not find a package configuration file provided by "dbw_mkz_msgs" with
  any of the following names:

    dbw_mkz_msgsConfig.cmake
    dbw_mkz_msgs-config.cmake

  Add the installation prefix of "dbw_mkz_msgs" to CMAKE_PREFIX_PATH or set
  "dbw_mkz_msgs_DIR" to a directory containing one of the above files.  If
  "dbw_mkz_msgs" provides a separate development package or SDK, be sure it
  has been installed.
Call Stack (most recent call first):
  styx/CMakeLists.txt:10 (find_package)


-- Could not find the required component 'dbw_mkz_msgs'. The following CMake error indicates that you either need to install the package with the same name or change your environment so that it can be found.
CMake Error at /opt/ros/kinetic/share/catkin/cmake/catkinConfig.cmake:83 (find_package):
  Could not find a package configuration file provided by "dbw_mkz_msgs" with
  any of the following names:

    dbw_mkz_msgsConfig.cmake
    dbw_mkz_msgs-config.cmake

  Add the installation prefix of "dbw_mkz_msgs" to CMAKE_PREFIX_PATH or set
  "dbw_mkz_msgs_DIR" to a directory containing one of the above files.  If
  "dbw_mkz_msgs" provides a separate development package or SDK, be sure it
  has been installed.
Call Stack (most recent call first):
  styx/CMakeLists.txt:10 (find_package)


-- Configuring incomplete, errors occurred!
See also "/home/workspace/CarND-Capstone/ros/build/CMakeFiles/CMakeOutput.log".
See also "/home/workspace/CarND-Capstone/ros/build/CMakeFiles/CMakeError.log".
Invoking "cmake" failed
```

run the following commands
```
sudo apt-get update
sudo apt-get install -y ros-kinetic-dbw-mkz-msgs
```
