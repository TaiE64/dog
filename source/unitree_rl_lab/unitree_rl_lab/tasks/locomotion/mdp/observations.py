from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
ALaM privileged observations (training only).
"""


def cof_velocity(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    command_name: str = "ee_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear velocity of centroid of LOCOMOTION feet (CoF) in world frame. (N, 3)

    Excludes the manipulation foot using leg_state from the command.
    Paper: CoF only includes locomotion legs (ℓ_i = 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, feet_cfg.body_ids, :]  # (N, num_feet, 3)
    # Get leg_state to mask out manipulation foot
    try:
        command = env.command_manager.get_command(command_name)
        leg_state = command[:, 6:10]  # (N, 4) — 1=loco, 0=manip
        # Weight each foot's velocity by its leg_state
        weights = leg_state.unsqueeze(-1)  # (N, 4, 1)
        weighted_vel = feet_vel * weights  # (N, 4, 3)
        num_loco_feet = leg_state.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1)  # (N, 1, 1)
        return weighted_vel.sum(dim=1) / num_loco_feet.squeeze(-1)  # (N, 3)
    except Exception:
        # Fallback: average all feet (before command manager is ready)
        return feet_vel.mean(dim=1)


def com_pos_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Center of mass position in the robot base frame. (N, 3)"""
    asset: Articulation = env.scene[asset_cfg.name]
    com_w = asset.data.root_com_pos_w  # (N, 3)
    com_rel = com_w - asset.data.root_pos_w
    return quat_apply_inverse(asset.data.root_quat_w, com_rel)  # (N, 3)


"""
ALaM privileged observations: domain randomization params (Table 1).
"""


def domain_rand_params(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Domain randomization parameters R (paper Table 1).

    Returns: total mass (1), CoM offset from default (3), friction proxy (1).
    These are available in sim but not on real robot — privileged info.
    Uses compact representation instead of per-body masses.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    # Total body mass (reflects payload randomization)
    total_mass = asset.data.default_mass.sum(dim=-1, keepdim=True).to(device)  # (N, 1)
    # CoM offset from default (reflects mass distribution changes)
    com_w = asset.data.root_com_pos_w  # (N, 3)
    com_default = asset.data.root_pos_w  # (N, 3) approximate
    com_offset = (com_w - com_default).to(device)  # (N, 3)
    # Friction proxy (placeholder — actual per-env friction is hard to query)
    friction = torch.ones(env.num_envs, 1, device=device)
    return torch.cat([total_mass, com_offset, friction], dim=-1)  # (N, 5)


"""
ALaM proprioceptive observations (Table 1).
"""


def foot_contact_indicator(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Foot contact indicator b ∈ R^4 (paper Table 1). 1=contact, 0=no contact. (N, num_feet)"""
    from isaaclab.sensors import ContactSensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0).float()


def foot_pos_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Foot positions in base frame r_a (paper Table 1). (N, num_feet * 3)"""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, num_feet, 3)
    root_pos = asset.data.root_pos_w.unsqueeze(1)  # (N, 1, 3)
    feet_rel = feet_pos_w - root_pos
    # Transform each foot to base frame
    N, num_feet, _ = feet_rel.shape
    feet_pos_b = torch.zeros_like(feet_rel)
    for i in range(num_feet):
        feet_pos_b[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, feet_rel[:, i, :])
    return feet_pos_b.reshape(N, -1)  # (N, num_feet * 3)


"""
Gait observations.
"""


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
