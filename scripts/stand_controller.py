#!/usr/bin/env python3
"""
Stand controller node for Go2 + DIY leg.
Waits for controllers to be ready, sends standing pose,
then unpauses Gazebo so the robot doesn't fall before control is active.
"""
import rospy
import yaml
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from controller_manager_msgs.srv import ListControllers

GO2_JOINTS = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

DIY_JOINTS = [
    'FL_diy_joint1', 'FL_diy_joint2', 'FL_diy_joint3', 'FL_diy_joint4',
    'diy_joint1', 'diy_joint2', 'diy_joint3', 'diy_joint4',
]

ALL_JOINTS = GO2_JOINTS + DIY_JOINTS


def joint_to_controller(joint_name):
    """Map joint name to controller name."""
    if joint_name in GO2_JOINTS:
        return joint_name.replace('_joint', '_controller')
    else:
        return joint_name + '_controller'


def wait_for_controllers():
    """Wait until all joint controllers are running."""
    rospy.loginfo('Waiting for controllers to be ready...')
    rospy.wait_for_service('/controller_manager/list_controllers')
    list_ctrl = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)

    expected = set(joint_to_controller(j) for j in ALL_JOINTS)
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        resp = list_ctrl()
        running = set(c.name for c in resp.controller if c.state == 'running')
        if expected.issubset(running):
            rospy.loginfo('All controllers are running.')
            return True
        rate.sleep()
    return False


def unpause_gazebo():
    """Unpause Gazebo physics."""
    rospy.loginfo('Unpausing Gazebo...')
    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    unpause()
    rospy.loginfo('Gazebo unpaused.')


def main():
    rospy.init_node('stand_controller')

    config_file = rospy.get_param('~config_file', '')
    if not config_file:
        rospy.logfatal('No config_file parameter provided')
        return

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    angles = config.get('default_joint_angles', {})

    # Create publishers
    pubs = {}
    for joint in ALL_JOINTS:
        ctrl_name = joint_to_controller(joint)
        topic = '/{}/command'.format(ctrl_name)
        pubs[joint] = rospy.Publisher(topic, Float64, queue_size=1)

    # Wait for all controllers to be running
    wait_for_controllers()

    # Pre-publish commands before unpausing (fills the topic buffer)
    for joint in ALL_JOINTS:
        pubs[joint].publish(Float64(angles.get(joint, 0.0)))
    rospy.sleep(0.5)

    # Now unpause - robot starts falling but controllers are already active
    unpause_gazebo()

    rospy.loginfo('Standing up: publishing joint angles')
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        for joint in ALL_JOINTS:
            pubs[joint].publish(Float64(angles.get(joint, 0.0)))
        rate.sleep()


if __name__ == '__main__':
    main()
