<?xml version="1.0"?>
<launch>

    <!-- Consumed topics -->
    <arg name="image"/>

    <!-- Produced topics -->
    <arg name="aruco_pose"      default="aruco_pose"/>

    <!-- Auxiliary -->
    <arg name="aruco_dict"      default="DICT_4X4_250"/>
    <arg name="aruco_size"      default="0.1"/>
    <arg name="aruco_tf_name"   default="aruco"/>


    <!-- Nodes -->

    <node name="aruco_detect" pkg="svea_vision" type="aruco_detect.py" output="screen">
        <!-- Topics -->
        <param name="sub_image"         value="$(arg image)"/>
        <param name="pub_aruco_pose"    value="$(arg aruco_pose)"/>
        <!-- Auxiliary -->
        <param name="aruco_dict_name"   value="$(arg aruco_dict)"/>
        <param name="aruco_size"        value="$(arg aruco_size)"/>
        <param name="aruco_tf_name"     value="$(arg aruco_tf_name)"/>
    </node>

</launch>
