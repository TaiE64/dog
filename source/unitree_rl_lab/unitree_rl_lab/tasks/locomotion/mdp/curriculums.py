from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


_diy_last_upgrade_step = -1


def diy_randomization_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str = "randomize_diy_joints",
    max_position_range: float = 1.5,
    step_size: float = 0.1,
    upgrade_interval: int = 1000,
) -> torch.Tensor:
    """Curriculum that increases diy joint randomization range every N steps."""
    global _diy_last_upgrade_step
    event_term = env.event_manager.get_term_cfg(event_term_name)
    current_range = event_term.params["position_range"][1]

    if _diy_last_upgrade_step < 0:
        _diy_last_upgrade_step = env.common_step_counter

    if env.common_step_counter - _diy_last_upgrade_step >= upgrade_interval:
        if current_range < max_position_range:
            new_range = min(current_range + step_size, max_position_range)
            event_term.params["position_range"] = (-new_range, new_range)
        _diy_last_upgrade_step = env.common_step_counter

    current_range = event_term.params["position_range"][1]
    return torch.tensor(current_range, device=env.device)


def manipulation_reward_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "ee_tracking",
    max_weight: float = 8.0,
    step_size: float = 0.5,
    upgrade_interval: int = 2000,
    stability_term_name: str = "alive",
    stability_threshold: float = 0.3,
    command_name: str = "ee_goal",
    max_manip_ratio: float = 1.0,
    manip_ratio_step: float = 0.1,
) -> torch.Tensor:
    """Gradually increase manipulation difficulty as locomotion stabilizes.

    Ramps up:
    1. ee_tracking reward weight: 0 → max_weight
    2. ee_raw_tracking reward weight: 0 → max_weight * 0.5
    3. rel_manip_envs: fraction of 3-legged envs → max_manip_ratio
    """
    # Use env-local state instead of module globals
    if not hasattr(env, "_manip_curriculum_last_step"):
        env._manip_curriculum_last_step = env.common_step_counter

    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    current_weight = reward_cfg.weight
    command_term = env.command_manager.get_term(command_name)

    if env.common_step_counter - env._manip_curriculum_last_step >= upgrade_interval:
        stability_reward = torch.mean(
            env.reward_manager._episode_sums[stability_term_name][env_ids]
        ) / env.max_episode_length_s
        stability_cfg = env.reward_manager.get_term_cfg(stability_term_name)

        if stability_reward > stability_cfg.weight * stability_threshold:
            # Ramp ee_tracking weight
            new_weight = min(current_weight + step_size, max_weight)
            reward_cfg.weight = new_weight
            # Also ramp raw goal tracking if it exists
            if "ee_raw_tracking" in env.reward_manager.active_terms:
                raw_cfg = env.reward_manager.get_term_cfg("ee_raw_tracking")
                raw_cfg.weight = min(raw_cfg.weight + step_size * 0.5, max_weight * 0.5)
            # Ramp manipulation env ratio toward 100%
            new_ratio = min(command_term.cfg.rel_manip_envs + manip_ratio_step, max_manip_ratio)
            command_term.cfg.rel_manip_envs = new_ratio

        env._manip_curriculum_last_step = env.common_step_counter

    return torch.tensor(reward_cfg.weight, device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
