"""Custom event terms for locomotion tasks."""

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


def randomize_joint_position_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set random joint position targets within the joint limits.

    The PD controller will smoothly drive the joints toward these targets.
    The joints move at a speed determined by the actuator's stiffness and damping.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the event to.
        position_range: The range of random offsets to add to the default joint positions.
            The result is clamped to soft joint limits.
        asset_cfg: The asset configuration to apply the event to.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # sample random target positions: default + random offset, clamped to limits
    joint_pos_target = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_pos_target += sample_uniform(*position_range, joint_pos_target.shape, joint_pos_target.device)

    # clamp to soft joint limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos_target = joint_pos_target.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # set as position target — PD controller will smoothly move joints there
    asset.set_joint_position_target(joint_pos_target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
