"""Velocity command directed toward the EE goal.

Like UniformVelocityCommand, but the velocity direction points toward
the EE goal position. Speed magnitude is still randomly sampled.
When the goal is close (<threshold), commands zero velocity (stand still).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GoalDirectedVelocityCommand(CommandTerm):
    """Velocity command pointing toward the EE goal.

    Command tensor: (num_envs, 3) = [vx, vy, wz] in base frame.
    Direction is toward the goal, speed is randomly sampled.
    """

    cfg: "GoalDirectedVelocityCommandCfg"

    def __init__(self, cfg: "GoalDirectedVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # Command buffer: [vx, vy, wz]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        # Sampled speed magnitude (persistent between updates)
        self._speed = torch.zeros(self.num_envs, device=self.device)
        # Metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "GoalDirectedVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tSpeed range: {self.cfg.ranges.speed}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """[vx, vy, wz] in base frame. Shape (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        vel_b = self.robot.data.root_lin_vel_b[:, :2]
        self.metrics["error_vel_xy"] = torch.norm(vel_b - self.vel_command_b[:, :2], dim=-1)
        ang_vel = self.robot.data.root_ang_vel_b[:, 2]
        self.metrics["error_vel_yaw"] = torch.abs(ang_vel - self.vel_command_b[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample random speed magnitude."""
        r = torch.empty(len(env_ids), device=self.device)
        self._speed[env_ids] = r.uniform_(*self.cfg.ranges.speed)

    def _update_command(self):
        """Every step: compute velocity direction toward goal in body frame."""
        # Get goal position G_d from the ee_goal command
        try:
            ee_cmd = self._env.command_manager.get_command(self.cfg.goal_command_name)
            G_d = ee_cmd[:, 0:3]  # raw goal in body frame
        except Exception:
            self.vel_command_b[:] = 0
            return

        # Direction to goal in xy plane
        goal_xy = G_d[:, :2]  # (N, 2)
        goal_dist = torch.norm(goal_xy, dim=-1)  # (N,)

        # Compute unit direction
        safe_dist = goal_dist.clamp(min=1e-6).unsqueeze(-1)  # (N, 1)
        goal_dir = goal_xy / safe_dist  # (N, 2) unit vector

        # Velocity = speed * direction (toward goal)
        speed = self._speed.unsqueeze(-1)  # (N, 1)
        self.vel_command_b[:, :2] = goal_dir * speed

        # Yaw: turn toward goal
        yaw_to_goal = torch.atan2(goal_xy[:, 1], goal_xy[:, 0])  # (N,)
        self.vel_command_b[:, 2] = yaw_to_goal.clamp(-1.0, 1.0)

        # Stand still when close to goal (< threshold)
        close = (goal_dist < self.cfg.stand_threshold).unsqueeze(-1)  # (N, 1)
        self.vel_command_b = torch.where(close, torch.zeros_like(self.vel_command_b), self.vel_command_b)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class GoalDirectedVelocityCommandCfg(CommandTermCfg):
    """Config for goal-directed velocity command."""

    class_type: type = GoalDirectedVelocityCommand

    asset_name: str = MISSING
    """Robot asset name."""

    goal_command_name: str = "ee_goal"
    """Name of the GoalPositionCommand to read goal from."""

    stand_threshold: float = 0.5
    """Distance (m) below which the robot should stand still and reach."""

    @configclass
    class Ranges:
        speed: tuple[float, float] = (0.0, 0.5)
        """Speed magnitude range (m/s). Expanded by velocity curriculum."""

    ranges: Ranges = Ranges()
