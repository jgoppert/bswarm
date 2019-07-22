#!/usr/bin/env python  
import roslib
import rospy

import tf
import nav_msgs.msg

def handle_pose(msg):
    # rospy.loginfo('received message')
    br = tf.TransformBroadcaster()
    pose = msg.pose.pose
    br.sendTransform(
        (pose.position.x, pose.position.y, pose.position.z),
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w) ,
        rospy.Time.now(),
        "base_link",
        "map")

if __name__ == '__main__':
    rospy.init_node('tf_gazebo_pose')
    rospy.Subscriber('/ground_truth',
                     nav_msgs.msg.Odometry,
                     handle_pose)
    rospy.spin()
