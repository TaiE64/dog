#!/usr/bin/env python3
"""
Bridge between MoveIt's FollowJointTrajectory action and
individual JointPositionController command topics.

MoveIt sends a trajectory via FollowJointTrajectory action.
This node interpolates the trajectory and publishes each joint's
target angle to its JointPositionController at 50Hz.
"""
import rospy
import actionlib
import numpy as np
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult
from std_msgs.msg import Float64


class ArmTrajectoryBridge:
    def __init__(self, arm_name, joint_names, default_positions):
        self.arm_name = arm_name
        self.joint_names = joint_names
        self.default_positions = default_positions
        self.current_targets = list(default_positions)
        self.executing = False

        # Publishers to individual JointPositionControllers
        self.pubs = {}
        for name in joint_names:
            self.pubs[name] = rospy.Publisher(
                f'/{name}_controller/command', Float64, queue_size=1)

        # Action server (MoveIt connects here)
        self.server = actionlib.SimpleActionServer(
            f'/{arm_name}_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction,
            execute_cb=self.execute_cb,
            auto_start=False)
        self.server.start()
        rospy.loginfo(f'{arm_name} trajectory bridge ready')

    def publish_current(self):
        """Publish current target positions (called from main loop)."""
        if not self.executing:
            for i, name in enumerate(self.joint_names):
                self.pubs[name].publish(Float64(self.current_targets[i]))

    def execute_cb(self, goal):
        """Execute a trajectory from MoveIt."""
        self.executing = True
        trajectory = goal.trajectory
        joint_names = trajectory.joint_names
        points = trajectory.points

        if not points:
            self.executing = False
            self.server.set_succeeded(FollowJointTrajectoryResult())
            return

        rospy.loginfo(f'{self.arm_name}: Executing trajectory with {len(points)} points')

        rate = rospy.Rate(50)
        start_time = rospy.Time.now()

        for i, point in enumerate(points):
            # Wait until this point's time
            target_time = start_time + point.time_from_start
            while rospy.Time.now() < target_time and not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    self.server.set_preempted()
                    return
                # Interpolate between previous and current point
                if i > 0:
                    prev_time = start_time + points[i-1].time_from_start
                    t_total = (target_time - prev_time).to_sec()
                    t_elapsed = (rospy.Time.now() - prev_time).to_sec()
                    if t_total > 0:
                        alpha = min(t_elapsed / t_total, 1.0)
                    else:
                        alpha = 1.0
                    for j, jname in enumerate(joint_names):
                        if jname in self.pubs:
                            pos = (1-alpha) * points[i-1].positions[j] + alpha * point.positions[j]
                            self.pubs[jname].publish(Float64(pos))
                else:
                    # First point: just publish directly
                    for j, jname in enumerate(joint_names):
                        if jname in self.pubs:
                            self.pubs[jname].publish(Float64(point.positions[j]))
                rate.sleep()

            # Publish final position for this point
            for j, jname in enumerate(joint_names):
                if jname in self.pubs:
                    self.pubs[jname].publish(Float64(point.positions[j]))

        # Hold final position for a moment
        for _ in range(10):
            for j, jname in enumerate(joint_names):
                if jname in self.pubs:
                    self.pubs[jname].publish(Float64(points[-1].positions[j]))
            rate.sleep()

        # Update current targets to final position
        for j, jname in enumerate(joint_names):
            if jname in self.joint_names:
                idx = self.joint_names.index(jname)
                self.current_targets[idx] = points[-1].positions[j]

        rospy.loginfo(f'{self.arm_name}: Trajectory complete')
        self.executing = False
        self.server.set_succeeded(FollowJointTrajectoryResult())


def main():
    rospy.init_node('arm_trajectory_bridge')

    fl_arm_bridge = ArmTrajectoryBridge(
        'fl_arm',
        ['FL_diy_joint1', 'FL_diy_joint2', 'FL_diy_joint3'],
        [0.1, 0.0, 0.0])

    fl_gripper_bridge = ArmTrajectoryBridge(
        'fl_gripper',
        ['FL_diy_joint4'],
        [-0.17])

    fr_arm_bridge = ArmTrajectoryBridge(
        'fr_arm',
        ['diy_joint1', 'diy_joint2', 'diy_joint3'],
        [0.1, 0.0, 0.0])

    fr_gripper_bridge = ArmTrajectoryBridge(
        'fr_gripper',
        ['diy_joint4'],
        [-0.17])

    # Continuously publish current targets at 50Hz
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        fl_arm_bridge.publish_current()
        fl_gripper_bridge.publish_current()
        fr_arm_bridge.publish_current()
        fr_gripper_bridge.publish_current()
        rate.sleep()


if __name__ == '__main__':
    main()
