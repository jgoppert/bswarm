#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import tf
import nav_msgs.msg


class CarController:


    def __init__(self):
        rospy.init_node('mycar_controller')
        self.time_last_plan = rospy.Time.now()
        self.pub_rear_left = rospy.Publisher('/mycar/force_rear_left', Wrench, queue_size=10)
        self.pub_rear_right = rospy.Publisher('/mycar/force_rear_right', Wrench, queue_size=10)
        self.pub_front_left = rospy.Publisher('/mycar/force_front_left', Wrench, queue_size=10)
        self.pub_front_right = rospy.Publisher('/mycar/force_front_right', Wrench, queue_size=10)
        self.sub_ground_truth = rospy.Subscriber('/mycar/ground_truth', Odometry,
                self.control_callback)
        self.sub_laser = rospy.Subscriber('/mycar/laser/scan', LaserScan,
                self.planning_callback)

    def rrt_planner(self):
        #TODO
        pass

    def planning_callback(self, laser_scan):
        now = rospy.Time.now()

        # find closest point for basic obstacle avoidance
        a0 = laser_scan.angle_min
        da = laser_scan.angle_increment
        i_min = np.argmin(laser_scan.ranges)
        closest_angle = a0 + da*i_min
        min_dist = laser_scan.ranges[i_min]
        # rospy.loginfo('min dist %f m %f deg', min_dist, np.rad2deg(closest_angle))

        # call long-term planning every 5 seconds
        if (now - self.time_last_plan).to_sec() > 5:
            self.time_last_plan = now
            rospy.loginfo('planning using RRT')

    def control_callback(self, odom):
        # position
        position = odom.pose.pose.position

        # orientation
        orientation = odom.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion(
            (orientation.x, orientation.y, orientation.z, orientation.w))
        yaw = euler[2]

        # rotation rate
        rates = odom.twist.twist.angular
        vel = odom.twist.twist.linear
        yaw_rate = rates.z

        # publish fram for transform tree
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (position.x, position.y, position.z),
            (orientation.x, orientation.y, orientation.z, orientation.w) ,
            rospy.Time.now(),
            "base_link",
            "odom")

        #rospy.loginfo('position %f %f %f, m', position.x, position.y, position.z)
        #rospy.loginfo('yaw %f, deg', np.rad2deg(yaw))
        #rospy.loginfo('yaw  rate %f, deg/s', np.rad2deg(yaw_rate))

        drive_torque = 0.1
        desired_yaw = np.deg2rad(90)

        # steering PD controller
        steer_kP = 1
        steer_kD = 1
        yaw_error = desired_yaw - yaw
        #rospy.loginfo("yaw error %f deg", np.rad2deg(yaw_error))
        yaw_rate_error = 0 - yaw_rate
        steer_torque = steer_kP*yaw_error + steer_kD*yaw_rate_error

        # speed P controller
        speed_kP = 1
        speed = np.sqrt(vel.x**2 + vel.y**2)
        desired_speed = 0.1
        speed_error = desired_speed - speed
        drive_torque = speed_kP*speed_error
        #rospy.loginfo("speed error %f m/s", speed_error)

        # saturate torques
        if np.abs(steer_torque) > 0.1:
            steer_torque /= np.abs(steer_torque)
        if np.abs(drive_torque) > 0.1:
            drive_torque /= np.abs(drive_torque)

        # rear drive
        msg = Wrench()
        msg.force.x = 0
        msg.force.y = 0
        msg.force.z = 0
        msg.torque.x = -drive_torque
        msg.torque.y = 0
        msg.torque.z = 0
        self.pub_rear_left.publish(msg)
        self.pub_rear_right.publish(msg)

        # front left steering
        msg = Wrench()
        msg.force.x = 0
        msg.force.y = 0
        msg.force.z = 0
        msg.torque.x = 0
        msg.torque.y = 0
        msg.torque.z = steer_torque
        self.pub_front_left.publish(msg)

        # front right steering
        msg = Wrench()
        msg.force.x = 0
        msg.force.y = 0
        msg.force.z = 0
        msg.torque.x = 0
        msg.torque.y = 0
        msg.torque.z = steer_torque
        self.pub_front_right.publish(msg)


if __name__ == "__main__":
    CarController()
    rospy.spin()
