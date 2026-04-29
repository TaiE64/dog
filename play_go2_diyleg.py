"""Sim2Sim: Deploy trained Go2+DIY-leg walk policy in MuJoCo viewer.

Usage:
    Terminal 1:  python play_go2_diyleg.py
    Terminal 2:  python keyboard_control.py
"""

import os
import socket
import threading
import numpy as np
import mujoco
import mujoco.viewer as viewer

try:
    import onnxruntime as rt
except ImportError:
    raise ImportError("pip install onnxruntime")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_PATH = os.path.join(_HERE, "leg_manipulator", "mjcf", "go2_diyleg.xml")
_ONNX_PATH = os.path.join(_HERE, "walk_model", "go2_diyleg_walk_policy_6200.onnx")

UDP_PORT = 9871

# Isaac Lab joint order for the 12 policy joints
ISAAC_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# Default joint angle offsets (Isaac Lab order: FL FR RL RR per joint group).
# Must match init_state.joint_pos in UNITREE_GO2_DIY_LEG_CFG (assets/robots/unitree.py).
# Note: front calves are POSITIVE (DIY base_link orientation), rear calves negative.
DEFAULT_OFFSETS = np.array([
    0.0,    0.0,    0.0,    0.0,     # hip:   FL FR RL RR
    0.560,  0.560,  0.401,  0.401,   # thigh: FL FR RL RR
    1.466,  1.466, -1.215, -1.215,   # calf:  FL FR RL RR
], dtype=np.float32)

ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05
ACTION_SCALE = 0.25


class Go2DiyLegController:
    """ONNX controller for Go2 + DIY-leg robot in MuJoCo."""

    def __init__(self, model: mujoco.MjModel, policy_path: str):
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )
        self._output_names = ["continuous_actions"]

        self._qpos_indices = np.array([
            model.joint(name).qposadr.item() for name in ISAAC_JOINT_NAMES
        ])
        self._qvel_indices = np.array([
            model.joint(name).dofadr.item() for name in ISAAC_JOINT_NAMES
        ])
        self._actuator_indices = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_actuator")
            for name in ISAAC_JOINT_NAMES
        ])

        self._last_action = np.zeros(12, dtype=np.float32)
        self._n_substeps = int(round(0.02 / model.opt.timestep))
        self._counter = self._n_substeps - 1  # run policy on first call
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self._imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")

        # DIY joints: random motion within joint limits
        DIY_JOINT_NAMES = [
            "FL_diy_joint1", "FL_diy_joint2", "FL_diy_joint3", "FL_diy_joint4",
            "diy_joint1", "diy_joint2", "diy_joint3", "diy_joint4",
        ]
        self._diy_act_ids = []
        self._diy_ranges = []
        for name in DIY_JOINT_NAMES:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_actuator")
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if act_id >= 0 and jnt_id >= 0:
                self._diy_act_ids.append(act_id)
                self._diy_ranges.append(model.jnt_range[jnt_id].copy())
        self._diy_targets = np.array([0.5 * (r[0] + r[1]) for r in self._diy_ranges])
        self._diy_interval = int(round(np.random.uniform(3.0, 8.0) / model.opt.timestep))
        self._diy_step = 0

        # Start UDP listener for keyboard commands
        self._start_udp_listener()

        print(f"Controller: {self._n_substeps} substeps, listening on UDP {UDP_PORT}")
        print(f"DIY joints: {len(self._diy_act_ids)} joints, random motion every 3-8s")

    def _start_udp_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", UDP_PORT))
        sock.settimeout(0.1)

        def listen():
            while True:
                try:
                    data, _ = sock.recvfrom(64)
                    parts = data.decode().strip().split()
                    if len(parts) == 3:
                        self.cmd_vel[0] = float(parts[0])
                        self.cmd_vel[1] = float(parts[1])
                        self.cmd_vel[2] = float(parts[2])
                except socket.timeout:
                    pass
                except Exception:
                    break

        t = threading.Thread(target=listen, daemon=True)
        t.start()

    def get_obs(self, model, data):
        gyro = data.sensor("gyro").data.copy()
        imu_xmat = data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_pos = data.qpos[self._qpos_indices] - DEFAULT_OFFSETS
        joint_vel = data.qvel[self._qvel_indices] * JOINT_VEL_SCALE
        return np.concatenate([
            gyro * ANG_VEL_SCALE, gravity, self.cmd_vel,
            np.clip(joint_pos, -100, 100),
            np.clip(joint_vel, -100, 100),
            self._last_action,
        ]).astype(np.float32)

    def get_control(self, model, data):
        self._counter += 1
        self._diy_step += 1

        # DIY random motion: pick new random targets every 3-8s
        if self._diy_step >= self._diy_interval:
            self._diy_step = 0
            self._diy_interval = int(round(np.random.uniform(3.0, 8.0) / model.opt.timestep))
            for i, r in enumerate(self._diy_ranges):
                self._diy_targets[i] = np.random.uniform(r[0], r[1])

        # Smoothly drive DIY joints to targets (PD actuators handle the smoothing)
        for i, act_id in enumerate(self._diy_act_ids):
            data.ctrl[act_id] = self._diy_targets[i]

        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            action = self._policy.run(self._output_names, {"obs": obs.reshape(1, -1)})[0][0]
            self._last_action = action.copy()
            targets = action * ACTION_SCALE + DEFAULT_OFFSETS
            for i in range(12):
                data.ctrl[self._actuator_indices[i]] = targets[i]


def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(_MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    controller = Go2DiyLegController(model, _ONNX_PATH)
    controller.cmd_vel[0] = 0.5  # default forward walk (policy can't stand still)

    # Initialize ctrl to default joint positions so robot doesn't fall before policy runs
    for i in range(12):
        data.ctrl[controller._actuator_indices[i]] = DEFAULT_OFFSETS[i]
    mujoco.set_mjcb_control(controller.get_control)

    return model, data


if __name__ == "__main__":
    print("=" * 60)
    print("  Go2 + DIY-Leg Sim2Sim in MuJoCo")
    print("=" * 60)
    print(f"MJCF: {_MJCF_PATH}")
    print(f"ONNX: {_ONNX_PATH}")
    print()
    print("Open another terminal and run:")
    print("  python keyboard_control.py")
    print()
    viewer.launch(loader=load_callback)
