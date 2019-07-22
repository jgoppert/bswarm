#!/usr/bin/env python  
import roslib
import rospy
import math
import numpy as np
import tf
import geometry_msgs.msg
import turtlesim.srv
import turtlesim.msg

class TurtleController:

    def __init__(self, i, init_time):
        self.i = i
        self.pub = rospy.Publisher('turtle{:d}/cmd_vel'.format(i), geometry_msgs.msg.Twist,queue_size=1)
        self.sub = rospy.Subscriber('turtle{:d}/pose'.format(i), turtlesim.msg.Pose, self.pose_callback, queue_size=1)
        self.s = 1
        self.last_time = rospy.Time.now()
        self.init_time = init_time
        
    
    def reference(self, t):
        """
        Returns position, velocity, accel of reference trajectory.
        """

        if t < 15:
            p = 2*np.array([np.cos(t), np.sin(t)]) + np.array([5, 5])
            v = 2*np.array([-np.sin(t), np.cos(t)])
            a = 2*np.array([-np.cos(t), -np.sin(t)])
        elif t < 30:
            p = np.array([self.i*1, 5])
            v = np.array([0, 0])
            a = np.array([0, 0])
            a = np.array([0, 0])
        elif t < 40:
            p = np.array([5, self.i*1])
            v = np.array([0, 0])
            a = np.array([0, 0])
        else:
            p = 3*np.array([np.cos(t), np.sin(t)]) + np.array([5, 5])
            v = 3*np.array([-np.sin(t), np.cos(t)])
            a = 3*np.array([-np.cos(t), -np.sin(t)])
        return p, v, a
    
    def pose_callback(self, pose):
        """
        Compute the control output whenver a pose msg is received.
        """
        
        # timing
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        t = (now - self.init_time).to_sec()
        self.last_time = now

        phi = 2*np.pi*self.i/10  # phase angle for individual vehicles
        rp, rv, ra = self.reference(t + phi)

        zeta = 1
        wn = 1
        p = np.array([pose.x, pose.y])
        v = np.array([pose.linear_velocity*np.cos(pose.theta), pose.linear_velocity*np.sin(pose.theta)])
        a = ra + 2*zeta*wn*(rv - v) + wn**2*(rp - p)

        # euler integration of controller state
        self.s += (a[0]*np.cos(pose.theta) + a[1]*np.sin(pose.theta))*dt
        if np.abs(self.s) < 1e-3:
            omega = 0
        else:
            omega = (-a[0]*np.sin(pose.theta) + a[1]*np.cos(pose.theta))/self.s

        cmd = geometry_msgs.msg.Twist()
        cmd.angular.z = omega
        cmd.linear.x = self.s
        self.pub.publish(cmd)

if __name__ == '__main__':
    n_turtles = 10

    rospy.init_node('turtle_tf_listener')
    listener = tf.TransformListener()

    # delete turtle 1 since we want to respawn it in a new location
    rospy.wait_for_service('kill')
    killer = rospy.ServiceProxy('kill', turtlesim.srv.Kill)
    killer('turtle1')

    # spawn turtle swarm
    rospy.wait_for_service('spawn')
    spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
    controllers = []
    init_time = rospy.Time.now()
    for i in range(n_turtles):
        theta = i*2*np.pi/n_turtles
        spawner(i*10/n_turtles, 5, 0, 'turtle{:d}'.format(i))
        controllers.append(TurtleController(i, init_time))

    # spin, waiting for threads to finish
    rospy.spin()
