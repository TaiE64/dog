#!/usr/bin/env python3
"""
Walk controller: deploys trained RL policy in Gazebo (ROS1).
Based on the working MuJoCo walk_policy_node.py.
No joint reordering - policy uses URDF order directly.
"""
import rospy
import torch
import torch.nn as nn
import numpy as np
import yaml
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import Empty
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Twist

# Go2 joint names in URDF order (matches policy order)
GO2_JOINT_NAMES = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

# Default standing pose (URDF order)
DEFAULT_POS = np.array([
    0.0, 0.543, -2.07,   # FL
    0.0, 0.543, -2.07,   # FR
    0.0, 0.905, -1.78,   # RL
    0.0, 0.905, -1.78,   # RR
], dtype=np.float32)

ACTION_SCALE = 0.25

DIY_JOINTS = {
    'FL_diy_joint1': 0.1, 'FL_diy_joint2': 0.0, 'FL_diy_joint3': 0.0, 'FL_diy_joint4': 0.0,
    'diy_joint1': 0.1, 'diy_joint2': 0.0, 'diy_joint3': 0.0, 'diy_joint4': 0.0,
}

# Policy order: grouped by joint type (hip-hip-hip-hip, thigh-thigh-thigh-thigh, calf-calf-calf-calf)
# URDF order: grouped by leg (hip-thigh-calf, hip-thigh-calf, ...)
# policy_idx -> urdf_idx
POLICY_TO_URDF = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
# urdf_idx -> policy_idx
URDF_TO_POLICY = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

# Default pos in POLICY order (grouped by type)
DEFAULT_POS_POLICY = np.array([
    0.0, 0.0, 0.0, 0.0,           # hips
    0.543, 0.543, 0.905, 0.905,    # thighs
    -2.07, -2.07, -1.78, -1.78,   # calfs
], dtype=np.float32)


def reorder_urdf_to_policy(urdf_data):
    return np.array([urdf_data[POLICY_TO_URDF[i]] for i in range(12)], dtype=np.float32)


def reorder_policy_to_urdf(policy_data):
    urdf_data = np.zeros(12, dtype=np.float32)
    for i in range(12):
        urdf_data[POLICY_TO_URDF[i]] = policy_data[i]
    return urdf_data


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q = (x, y, z, w)."""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class WalkController:
    def __init__(self):
        rospy.init_node('walk_controller')

        # Load model
        model_path = rospy.get_param('~model_path', '')
        rospy.loginfo(f'Loading model from {model_path}')

        self.policy = nn.Sequential(
            nn.Linear(45, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        actor_state = {
            k.replace('actor.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('actor.')
        }
        self.policy.load_state_dict(actor_state)
        self.policy.eval()
        rospy.loginfo('Model loaded successfully')

        # State variables
        self.joint_pos = np.zeros(12, dtype=np.float32)
        self.joint_vel = np.zeros(12, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.quat = np.array([0, 0, 0, 1], dtype=np.float32)  # (x,y,z,w)
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.joint_name_to_idx = {}
        self.imu_ready = False
        self.joints_ready = False

        # Publishers (Go2 + DIY)
        self.pubs = {}
        for name in GO2_JOINT_NAMES:
            ctrl = name.replace('_joint', '_controller')
            self.pubs[name] = rospy.Publisher(f'/{ctrl}/command', Float64, queue_size=1)
        for name in DIY_JOINTS:
            self.pubs[name] = rospy.Publisher(f'/{name}_controller/command', Float64, queue_size=1)

        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_cb)

    def joint_state_cb(self, msg):
        if not self.joint_name_to_idx:
            self.joint_name_to_idx = {name: i for i, name in enumerate(msg.name)}
            rospy.loginfo(f'Joint states names: {list(msg.name)}')
        for i, name in enumerate(GO2_JOINT_NAMES):
            if name in self.joint_name_to_idx:
                idx = self.joint_name_to_idx[name]
                self.joint_pos[i] = msg.position[idx]
                self.joint_vel[i] = msg.velocity[idx]
        self.joints_ready = True

    def imu_cb(self, msg):
        self.ang_vel[0] = msg.angular_velocity.x
        self.ang_vel[1] = msg.angular_velocity.y
        self.ang_vel[2] = msg.angular_velocity.z
        self.quat[0] = msg.orientation.x
        self.quat[1] = msg.orientation.y
        self.quat[2] = msg.orientation.z
        self.quat[3] = msg.orientation.w
        self.imu_ready = True

    def cmd_vel_cb(self, msg):
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.linear.y
        self.cmd_vel[2] = msg.angular.z

    def build_obs(self):
        # ang_vel from Gazebo IMU is already in body frame, no need to rotate
        body_ang_vel = self.ang_vel
        # gravity needs to be projected from world to body frame
        projected_gravity = quat_rotate_inverse(self.quat, self.gravity_vec)

        # Reorder joint data from URDF to policy order
        joint_pos_policy = reorder_urdf_to_policy(self.joint_pos)
        joint_vel_policy = reorder_urdf_to_policy(self.joint_vel)
        joint_pos_rel = joint_pos_policy - DEFAULT_POS_POLICY

        obs = np.concatenate([
            body_ang_vel * 0.2,          # 3
            projected_gravity,            # 3
            self.cmd_vel,                 # 3
            joint_pos_rel,                # 12
            joint_vel_policy * 0.05,      # 12
            self.last_action,             # 12
        ])
        return obs  # 45 dim

    def run(self):
        # Wait for controllers
        rospy.loginfo('Waiting for controllers...')
        rospy.wait_for_service('/controller_manager/list_controllers')
        list_ctrl = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
        expected = set(j.replace('_joint', '_controller') for j in GO2_JOINT_NAMES)
        expected.update(f'{name}_controller' for name in DIY_JOINTS)
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            resp = list_ctrl()
            running = set(c.name for c in resp.controller if c.state == 'running')
            if expected.issubset(running):
                break
            rate.sleep()
        rospy.loginfo('All controllers running.')

        # Wait for IMU and joint states
        rospy.loginfo('Waiting for sensors...')
        while not (self.imu_ready and self.joints_ready) and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Unpause Gazebo
        rospy.loginfo('Unpausing Gazebo...')
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause()

        # Stand for 3 seconds
        rospy.loginfo('Standing for 3 seconds...')
        stand_rate = rospy.Rate(50)
        stand_start = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - stand_start).to_sec()
            if elapsed >= 3.0:
                break
            for i, name in enumerate(GO2_JOINT_NAMES):
                self.pubs[name].publish(Float64(DEFAULT_POS[i]))
            for name, pos in DIY_JOINTS.items():
                self.pubs[name].publish(Float64(pos))
            stand_rate.sleep()

        # Walk policy loop at 50 Hz
        rospy.loginfo('Starting walk policy.')
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            obs = self.build_obs()
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                action = self.policy(obs_tensor).squeeze(0).numpy()

            self.last_action = action.copy()

            # action is in policy order, convert to target pos then reorder to URDF
            target_policy = DEFAULT_POS_POLICY + action * ACTION_SCALE
            target_urdf = reorder_policy_to_urdf(target_policy)

            for i, name in enumerate(GO2_JOINT_NAMES):
                self.pubs[name].publish(Float64(target_urdf[i]))
            for name, pos in DIY_JOINTS.items():
                self.pubs[name].publish(Float64(pos))

            rate.sleep()


if __name__ == '__main__':
    try:
        controller = WalkController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
