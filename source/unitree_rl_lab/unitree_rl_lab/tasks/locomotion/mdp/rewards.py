from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


"""
ALaM rewards: gravitational moment minimization & manipulation.
Reference: "Robust Pedipulation on Quadruped Robots via Gravitational-moment Minimization"
"""


def gravitational_moment_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sigma: float = 10.0,
    command_name: str = "ee_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for minimizing the gravitational moment about the centroid of LOCOMOTION feet (CoF).

    Uses leg_state to exclude the manipulation foot from CoF calculation.
    Feet/leg_state order: [FL, FR, RL, RR] (matches URDF).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_pos = asset.data.root_com_pos_w  # (N, 3)
    feet_pos = asset.data.body_pos_w[:, feet_cfg.body_ids, :]  # (N, num_feet, 3)
    # Get leg_state to weight feet (exclude manipulation foot)
    try:
        command = env.command_manager.get_command(command_name)
        leg_state = command[:, 6:10]  # (N, 4) — 1=loco, 0=manip
        weights = leg_state.unsqueeze(-1)  # (N, 4, 1)
        weighted_pos = feet_pos[:, :, :2] * weights[:, :, :1].expand_as(feet_pos[:, :, :2])
        num_loco = leg_state.sum(dim=-1, keepdim=True).clamp(min=1)  # (N, 1)
        cof_xy = weighted_pos.sum(dim=1) / num_loco  # (N, 2)
    except Exception:
        cof_xy = feet_pos[:, :, :2].mean(dim=1)
    com_xy = com_pos[:, :2]
    moment_magnitude = torch.norm(com_xy - cof_xy, dim=-1)
    return torch.exp(-moment_magnitude / sigma)


def locomotion_feet_area_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sigma: float = 0.05,
    command_name: str = "ee_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maximizing the support polygon area of LOCOMOTION feet only.

    Uses the shoelace formula on locomotion feet projected to xy-plane.
    Manipulation foot is excluded via leg_state.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, feet_cfg.body_ids, :2]  # (N, num_feet, 2)
    # Compute centroid
    centroid = feet_pos.mean(dim=1, keepdim=True)  # (num_envs, 1, 2)
    # Sort by angle from centroid to form convex polygon
    diff = feet_pos - centroid  # (num_envs, num_feet, 2)
    angles = torch.atan2(diff[:, :, 1], diff[:, :, 0])  # (num_envs, num_feet)
    sorted_indices = torch.argsort(angles, dim=1)
    # Gather sorted positions
    sorted_feet = torch.gather(feet_pos, 1, sorted_indices.unsqueeze(-1).expand_as(feet_pos))
    # Shoelace formula for polygon area
    n = sorted_feet.shape[1]
    x = sorted_feet[:, :, 0]  # (num_envs, n)
    y = sorted_feet[:, :, 1]
    # sum(x_i * y_{i+1} - x_{i+1} * y_i)
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    area = 0.5 * torch.abs(torch.sum(x * y_next - x_next * y, dim=1))  # (num_envs,)
    return torch.clamp(torch.exp(area / sigma) - 1.0, max=1.0)


def _get_manip_ee_pos_b(env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper: get manipulation EE position in base frame. (N, 3)"""
    asset: Articulation = env.scene[ee_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, ee_cfg.body_ids, :]
    ee_pos_w_mean = ee_pos_w.mean(dim=1)
    ee_rel = ee_pos_w_mean - asset.data.root_pos_w
    return quat_apply_inverse(asset.data.root_quat_w, ee_rel)


def ee_goal_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_cfg: SceneEntityCfg,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking the PROJECTED goal G_B (paper Table 3).

    Command layout: [G_d(3), G_B(3), leg_state(4)].
    Tracks G_B (indices 3:6), the reachability-projected goal.
    """
    command = env.command_manager.get_command(command_name)
    G_B = command[:, 3:6]  # projected goal in base frame
    ee_pos_b = _get_manip_ee_pos_b(env, ee_cfg)
    error = torch.norm(ee_pos_b - G_B, dim=-1)
    return torch.exp(-error / sigma)


def ee_raw_goal_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_cfg: SceneEntityCfg,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking the RAW goal G_d (paper Table 3).

    Command layout: [G_d(3), G_B(3), leg_state(4)].
    Tracks G_d (indices 0:3), the un-projected goal. This reward keeps
    increasing as the robot walks closer to the goal.
    """
    command = env.command_manager.get_command(command_name)
    G_d = command[:, 0:3]  # raw goal in base frame
    ee_pos_b = _get_manip_ee_pos_b(env, ee_cfg)
    error = torch.norm(ee_pos_b - G_d, dim=-1)
    return torch.exp(-error / sigma)


def action_smoothness_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Second-order action smoothness penalty (paper Table 2, w11).

    Penalizes: ||a_t - 2*a_{t-1} + a_{t-2}||^2
    Resets history at episode boundaries.
    """
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if not hasattr(env, "_alam_prev_prev_action"):
        env._alam_prev_prev_action = torch.zeros_like(action)
    prev_prev = env._alam_prev_prev_action
    # Reset history at episode start (episode_length_buf == 0 means just reset)
    reset_mask = (env.episode_length_buf <= 1).unsqueeze(-1)
    prev_prev = torch.where(reset_mask, torch.zeros_like(prev_prev), prev_prev)
    smoothness = torch.sum(torch.square(action - 2 * prev + prev_prev), dim=-1)
    # Update history
    env._alam_prev_prev_action = prev.clone()
    return smoothness


def soft_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Soft contact reward (paper Table 2, w12).

    Rewards feet making gentle contact (low force magnitude).
    Returns sum of contact force magnitudes for locomotion feet.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, num_feet, 3)
    force_mag = torch.norm(forces, dim=-1)  # (N, num_feet)
    return torch.sum(force_mag, dim=-1)


def diy_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize high velocities on the DIY manipulation joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def diy_joint_acc_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize acceleration (jerk) on the DIY manipulation joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)


def diy_action_rate(
    env: ManagerBasedRLEnv,
    loco_action_dim: int = 12,
) -> torch.Tensor:
    """Penalize rate of change of the manipulation (DIY) portion of the action."""
    # Actions are [loco(12), manip(8)]; take the manip slice
    manip_actions = env.action_manager.action[:, loco_action_dim:]
    manip_prev = env.action_manager.prev_action[:, loco_action_dim:]
    return torch.sum(torch.square(manip_actions - manip_prev), dim=-1)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
