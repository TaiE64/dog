#!/usr/bin/env python3
"""
Walk controller: deploys trained RL policy (model_1700.pt) in Gazebo.
Reads IMU + joint_states, runs policy inference, publishes joint position commands.
"""
import rospy
import torch
import numpy as np
import yaml
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import Empty
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Twist

# Go2 joint names in URDF order
GO2_JOINT_NAMES = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

# Default standing pose (from deploy.yaml default_joint_pos, reordered to URDF order)
DEFAULT_POS = np.array([
    0.0, 0.543, -2.07,   # FL
    0.0, 0.543, -2.07,   # FR
    0.0, 0.905, -1.78,   # RL
    0.0, 0.905, -1.78,   # RR
], dtype=np.float32)

# Action scale and offset (from deploy.yaml)
ACTION_SCALE = np.array([0.25]*12, dtype=np.float32)
ACTION_OFFSET = np.array([
    0.0, 0.543, -2.07,
    0.0, 0.543, -2.07,
    0.0, 0.905, -1.78,
    0.0, 0.905, -1.78,
], dtype=np.float32)

# Observation scales (from deploy.yaml)
ANG_VEL_SCALE = 0.2
GRAVITY_SCALE = 1.0
CMD_SCALE = 1.0
JOINT_POS_SCALE = 1.0
JOINT_VEL_SCALE = 0.05
ACTION_OBS_SCALE = 1.0


class WalkController:
    def __init__(self):
        rospy.init_node('walk_controller')

        # Load deploy config
        deploy_path = rospy.get_param('~deploy_config', '')
        with open(deploy_path, 'r') as f:
            self.deploy_cfg = yaml.unsafe_load(f)

        # joint_ids_map: maps Isaac Lab joint order -> URDF joint order
        # First 12 entries are for Go2 legs
        self.joint_ids_map = self.deploy_cfg['joint_ids_map'][:12]

        # Load model
        model_path = rospy.get_param('~model_path', '')
        rospy.loginfo(f'Loading model from {model_path}')
        checkpoint = torch.load(model_path, map_location='cpu')

        # Build actor network from agent.yaml config
        actor_dims = [512, 256, 128]
        obs_dim = 3 + 3 + 3 + 12 + 12 + 12  # ang_vel + gravity + cmd + joint_pos + joint_vel + last_action = 45
        layers = []
        in_dim = obs_dim
        for hidden_dim in actor_dims:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 12))  # 12 actions for Go2 joints
        self.actor = torch.nn.Sequential(*layers)

        # Load weights from checkpoint
        model_state = checkpoint['model_state_dict']
        actor_state = {}
        layer_idx = 0
        for key in sorted(model_state.keys()):
            if 'actor' in key:
                # Map rsl_rl actor weights
                param_name = key.split('actor.')[-1]
                actor_state[param_name] = model_state[key]
        self.actor.load_state_dict(actor_state)
        self.actor.eval()
        rospy.loginfo('Model loaded successfully')

        # State variables
        self.joint_pos = np.zeros(12, dtype=np.float32)
        self.joint_vel = np.zeros(12, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.cmd_vel = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]
        self.last_action = np.zeros(12, dtype=np.float32)
        self.joint_state_received = False

        # Publishers
        self.pubs = {}
        for name in GO2_JOINT_NAMES:
            ctrl = name.replace('_joint', '_controller')
            self.pubs[name] = rospy.Publisher(f'/{ctrl}/command', Float64, queue_size=1)

        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_cb)

    def joint_state_cb(self, msg):
        """Map joint_states to our 12-joint array."""
        for i, name in enumerate(GO2_JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_pos[i] = msg.position[idx]
                self.joint_vel[i] = msg.velocity[idx]
        self.joint_state_received = True

    def imu_cb(self, msg):
        """Extract angular velocity and compute projected gravity from orientation."""
        self.base_ang_vel[0] = msg.angular_velocity.x
        self.base_ang_vel[1] = msg.angular_velocity.y
        self.base_ang_vel[2] = msg.angular_velocity.z

        # Compute projected gravity from quaternion
        q = msg.orientation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        # Rotate gravity vector [0,0,-1] by inverse of body quaternion
        self.projected_gravity[0] = 2.0 * (qx*qz - qw*qy)
        self.projected_gravity[1] = 2.0 * (qy*qz + qw*qx)
        self.projected_gravity[2] = 1.0 - 2.0 * (qx*qx + qy*qy)

    def cmd_vel_cb(self, msg):
        """Receive velocity commands."""
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.linear.y
        self.cmd_vel[2] = msg.angular.z

    def reorder_to_isaac(self, urdf_data):
        """Reorder from URDF order to Isaac Lab policy order using joint_ids_map."""
        isaac_data = np.zeros(12, dtype=np.float32)
        for isaac_idx in range(12):
            urdf_idx = self.joint_ids_map[isaac_idx]
            if urdf_idx < 12:
                isaac_data[isaac_idx] = urdf_data[urdf_idx]
        return isaac_data

    def reorder_to_urdf(self, isaac_data):
        """Reorder from Isaac Lab policy order back to URDF order."""
        urdf_data = np.zeros(12, dtype=np.float32)
        for isaac_idx in range(12):
            urdf_idx = self.joint_ids_map[isaac_idx]
            if urdf_idx < 12:
                urdf_data[urdf_idx] = isaac_data[isaac_idx]
        return urdf_data

    def build_obs(self):
        """Build observation vector in Isaac Lab order."""
        # Reorder joint data to Isaac Lab order
        joint_pos_isaac = self.reorder_to_isaac(self.joint_pos)
        joint_vel_isaac = self.reorder_to_isaac(self.joint_vel)
        default_pos_isaac = self.reorder_to_isaac(DEFAULT_POS)
        last_action_isaac = self.last_action  # already in Isaac order

        # Joint pos relative to default
        joint_pos_rel = joint_pos_isaac - default_pos_isaac

        obs = np.concatenate([
            self.base_ang_vel * ANG_VEL_SCALE,        # 3
            self.projected_gravity * GRAVITY_SCALE,     # 3
            self.cmd_vel * CMD_SCALE,                   # 3
            joint_pos_rel * JOINT_POS_SCALE,            # 12
            joint_vel_isaac * JOINT_VEL_SCALE,          # 12
            last_action_isaac * ACTION_OBS_SCALE,       # 12
        ])
        return obs  # 45 dim

    def run(self):
        # Wait for controllers
        rospy.loginfo('Waiting for controllers...')
        rospy.wait_for_service('/controller_manager/list_controllers')
        list_ctrl = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
        expected = set(j.replace('_joint', '_controller') for j in GO2_JOINT_NAMES)
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            resp = list_ctrl()
            running = set(c.name for c in resp.controller if c.state == 'running')
            if expected.issubset(running):
                break
            rate.sleep()
        rospy.loginfo('All controllers running.')

        # Wait for joint states
        rospy.loginfo('Waiting for joint states...')
        while not self.joint_state_received and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Unpause Gazebo
        rospy.loginfo('Unpausing Gazebo...')
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause()

        # Phase 1: Stand for 3 seconds using default joint angles
        rospy.loginfo('Standing up for 3 seconds...')
        stand_rate = rospy.Rate(50)
        stand_start = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - stand_start).to_sec()
            if elapsed >= 3.0:
                break
            for i, name in enumerate(GO2_JOINT_NAMES):
                self.pubs[name].publish(Float64(DEFAULT_POS[i]))
            stand_rate.sleep()

        rospy.loginfo('Starting walk policy.')

        # Phase 2: RL policy control loop at 50 Hz (step_dt = 0.02)
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # Build observation
            obs = self.build_obs()
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            # Policy inference
            with torch.no_grad():
                action = self.actor(obs_tensor).squeeze(0).numpy()

            # Store last action (in Isaac order)
            self.last_action = action.copy()

            # Convert action to target joint positions (Isaac order)
            # target = action * scale + offset
            offset_isaac = self.reorder_to_isaac(ACTION_OFFSET)
            scale_isaac = self.reorder_to_isaac(ACTION_SCALE)
            target_isaac = action * scale_isaac + offset_isaac

            # Reorder to URDF order
            target_urdf = self.reorder_to_urdf(target_isaac)

            # Publish
            for i, name in enumerate(GO2_JOINT_NAMES):
                self.pubs[name].publish(Float64(target_urdf[i]))

            rate.sleep()


if __name__ == '__main__':
    try:
        controller = WalkController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
