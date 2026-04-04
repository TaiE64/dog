#!/usr/bin/env python3
"""
PyBullet sim2sim: deploy trained RL walk policy with exact PD control matching Isaac Lab.
Usage: python3 pybullet_sim2sim.py
"""
import time
import numpy as np
import torch
import pybullet as p
import pybullet_data

# ======================== Config ========================
URDF_PATH = "/home/taie/dog_ws/src/leg_manipulator/urdf/go2_with_diy_leg.urdf"
MODEL_PATH = "/home/taie/dog_ws/walk_model/walk_model/model_1700.pt"

# Physics
SIM_DT = 0.005          # Physics step (matches Isaac Lab)
DECIMATION = 4           # Policy runs every 4 physics steps = 0.02s = 50Hz
POLICY_DT = SIM_DT * DECIMATION

# PD gains (from deploy.yaml, first 12 for Go2 joints)
KP = 25.0
KD = 0.5
MAX_TORQUE = 23.4  # UnitreeActuator Y2 (peak torque)

# Go2 joint names in URDF order
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

# Action scale and offset (URDF order, from deploy.yaml)
ACTION_SCALE = np.array([0.25]*12, dtype=np.float32)
ACTION_OFFSET = DEFAULT_POS.copy()

# Isaac Lab joint_ids_map (first 12): maps Isaac policy index -> URDF joint index
JOINT_IDS_MAP = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]

# Observation scales
ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05

# Velocity command
CMD_VEL = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # [vx, vy, wz]


def reorder_to_isaac(urdf_data):
    isaac_data = np.zeros(12, dtype=np.float32)
    for isaac_idx in range(12):
        urdf_idx = JOINT_IDS_MAP[isaac_idx]
        isaac_data[isaac_idx] = urdf_data[urdf_idx]
    return isaac_data


def reorder_to_urdf(isaac_data):
    urdf_data = np.zeros(12, dtype=np.float32)
    for isaac_idx in range(12):
        urdf_idx = JOINT_IDS_MAP[isaac_idx]
        urdf_data[urdf_idx] = isaac_data[isaac_idx]
    return urdf_data


def build_actor(obs_dim=45, action_dim=12):
    """Build actor network matching rsl_rl ActorCritic [512, 256, 128]."""
    layers = []
    dims = [obs_dim, 512, 256, 128, action_dim]
    for i in range(len(dims)-1):
        layers.append(torch.nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers)


def load_policy(model_path):
    """Load trained policy weights."""
    actor = build_actor()
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']

    actor_state = {}
    for key in model_state:
        if 'actor' in key:
            param_name = key.split('actor.')[-1]
            actor_state[param_name] = model_state[key]

    actor.load_state_dict(actor_state)
    actor.eval()
    print(f"Loaded policy from {model_path}")
    return actor


def get_gravity_vector(robot_id):
    """Get projected gravity vector in body frame."""
    _, quat = p.getBasePositionAndOrientation(robot_id)
    # PyBullet quaternion: [x, y, z, w]
    qx, qy, qz, qw = quat
    # Rotate gravity [0, 0, -1] by inverse of body quaternion
    gx = 2.0 * (qx*qz - qw*qy)
    gy = 2.0 * (qy*qz + qw*qx)
    gz = 1.0 - 2.0 * (qx*qx + qy*qy)
    return np.array([gx, gy, gz], dtype=np.float32)


def main():
    # ======================== Setup PyBullet ========================
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(SIM_DT)

    # Ground plane
    p.loadURDF("plane.urdf")

    # Load robot
    robot_id = p.loadURDF(
        URDF_PATH,
        basePosition=[0, 0, 0.32],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
    )

    # Map joint names to PyBullet joint indices
    joint_name_to_id = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode('utf-8')
        joint_name_to_id[name] = i

    go2_joint_ids = []
    for name in GO2_JOINT_NAMES:
        if name in joint_name_to_id:
            go2_joint_ids.append(joint_name_to_id[name])
        else:
            print(f"WARNING: Joint {name} not found in URDF!")
            go2_joint_ids.append(-1)

    print(f"Go2 joint IDs: {go2_joint_ids}")

    # Disable default motor control for Go2 joints (we apply torques manually)
    for jid in go2_joint_ids:
        if jid >= 0:
            p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL, force=0)

    # Set initial joint positions
    for i, jid in enumerate(go2_joint_ids):
        if jid >= 0:
            p.resetJointState(robot_id, jid, DEFAULT_POS[i])

    # Load policy
    actor = load_policy(MODEL_PATH)

    # ======================== Sim Loop ========================
    last_action = np.zeros(12, dtype=np.float32)
    step_count = 0

    # Stand for 2 seconds first
    print("Standing for 2 seconds...")
    stand_steps = int(2.0 / SIM_DT)
    for _ in range(stand_steps):
        # Get joint states
        joint_pos = np.zeros(12, dtype=np.float32)
        joint_vel = np.zeros(12, dtype=np.float32)
        for i, jid in enumerate(go2_joint_ids):
            if jid >= 0:
                state = p.getJointState(robot_id, jid)
                joint_pos[i] = state[0]
                joint_vel[i] = state[1]

        # PD torque to hold standing pose
        torques = KP * (DEFAULT_POS - joint_pos) - KD * joint_vel
        torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

        # Apply torques
        for i, jid in enumerate(go2_joint_ids):
            if jid >= 0:
                p.setJointMotorControl2(robot_id, jid, p.TORQUE_CONTROL, force=torques[i])

        p.stepSimulation()
        time.sleep(SIM_DT)

    print("Starting walk policy! Press Ctrl+C to stop.")
    print("Velocity command: vx=0.5 m/s")

    try:
        while True:
            # Get joint states
            joint_pos = np.zeros(12, dtype=np.float32)
            joint_vel = np.zeros(12, dtype=np.float32)
            for i, jid in enumerate(go2_joint_ids):
                if jid >= 0:
                    state = p.getJointState(robot_id, jid)
                    joint_pos[i] = state[0]
                    joint_vel[i] = state[1]

            # Run policy every DECIMATION steps
            if step_count % DECIMATION == 0:
                # Get base angular velocity (body frame)
                _, ang_vel = p.getBaseVelocity(robot_id)
                # Transform to body frame
                _, quat = p.getBasePositionAndOrientation(robot_id)
                # Inverse rotate world ang_vel to body frame
                inv_quat = [-quat[0], -quat[1], -quat[2], quat[3]]
                ang_vel_body = np.array(p.multiplyTransforms(
                    [0,0,0], inv_quat, ang_vel, [0,0,0,1])[0], dtype=np.float32)

                # Projected gravity
                gravity = get_gravity_vector(robot_id)

                # Build observation (Isaac order)
                joint_pos_isaac = reorder_to_isaac(joint_pos)
                joint_vel_isaac = reorder_to_isaac(joint_vel)
                default_pos_isaac = reorder_to_isaac(DEFAULT_POS)

                joint_pos_rel = joint_pos_isaac - default_pos_isaac

                obs = np.concatenate([
                    ang_vel_body * ANG_VEL_SCALE,       # 3
                    gravity,                              # 3
                    CMD_VEL,                              # 3
                    joint_pos_rel,                        # 12
                    joint_vel_isaac * JOINT_VEL_SCALE,    # 12
                    last_action,                          # 12
                ])

                # Policy inference
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_tensor).squeeze(0).numpy()

                last_action = action.copy()

                # Action -> target joint angles (Isaac order)
                scale_isaac = reorder_to_isaac(ACTION_SCALE)
                offset_isaac = reorder_to_isaac(ACTION_OFFSET)
                target_isaac = action * scale_isaac + offset_isaac

                # Back to URDF order
                target_pos = reorder_to_urdf(target_isaac)

            # PD torque
            torques = KP * (target_pos - joint_pos) - KD * joint_vel
            torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

            # Apply torques
            for i, jid in enumerate(go2_joint_ids):
                if jid >= 0:
                    p.setJointMotorControl2(robot_id, jid, p.TORQUE_CONTROL, force=torques[i])

            p.stepSimulation()
            step_count += 1
            time.sleep(SIM_DT)

    except KeyboardInterrupt:
        print("\nStopping.")

    p.disconnect()


if __name__ == '__main__':
    main()
