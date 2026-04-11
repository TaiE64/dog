"""ALaM goal position command generator.

Goals are sampled in WORLD frame (e.g. -5m to 5m) and transformed to
body frame every step. Reachability map projects each step.
Command output: [G_d(3), G_B(3), leg_state(4)] = 10 dims.
  - G_d: raw goal in body frame (for raw goal tracking reward)
  - G_B: projected goal in body frame (for projected goal tracking reward)
  - leg_state: [FR, FL, RR, RL], 1=loco, 0=manip
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


class GoalPositionCommand(CommandTerm):
    """World-frame goal positions with per-step body-frame transform and reachability map.

    Command tensor: (num_envs, 10) = [G_d(3), G_B(3), leg_state(4)].
    """

    cfg: "GoalPositionCommandCfg"

    def __init__(self, cfg: "GoalPositionCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Resolve body indices
        self._ee_body_ids = self.robot.find_bodies(cfg.ee_body_name)[0]
        self._fr_hip_body_id = self.robot.find_bodies("FR_hip")[0][0]
        self._fl_hip_body_id = self.robot.find_bodies("FL_hip")[0][0]

        # Resolve leg joint indices in the action space to determine
        # which index in leg_state corresponds to FR vs FL
        # This avoids hardcoding any URDF ordering assumptions.
        hip_ids, hip_names = self.robot.find_joints(".*_hip_joint")
        self._leg_order = [n.replace("_hip_joint", "") for n in hip_names]
        self._fr_leg_idx = self._leg_order.index("FR")
        self._fl_leg_idx = self._leg_order.index("FL")
        print(f"[GoalPositionCommand] Leg order from robot: {self._leg_order}")
        print(f"[GoalPositionCommand] FR leg index: {self._fr_leg_idx}, FL leg index: {self._fl_leg_idx}")

        # Goal in WORLD frame (persistent until resampled)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Command buffer: [G_d(3), G_B(3), leg_state(4)] = 10 dims
        # leg_state order matches the ACTUAL robot joint order (resolved above)
        self._command = torch.zeros(self.num_envs, 10, device=self.device)
        self._command[:, 6:] = 1.0  # all legs locomotion initially

        # Metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "GoalPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tReachability radius: {self.cfg.reachability_radius}\n"
        msg += f"\tManip env ratio: {self.cfg.rel_manip_envs}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """[G_d(3), G_B(3), leg_state(4)]. Shape (num_envs, 10)."""
        return self._command

    def _update_metrics(self):
        # EE position in base frame
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_ids, :]
        ee_pos_mean_w = ee_pos_w.mean(dim=1)
        ee_rel = ee_pos_mean_w - self.robot.data.root_pos_w
        ee_pos_b = quat_apply_inverse(self.robot.data.root_quat_w, ee_rel)
        # Error to projected goal G_B
        G_B = self._command[:, 3:6]
        self.metrics["position_error"] = torch.norm(ee_pos_b - G_B, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goals in WORLD frame and determine leg selection."""
        n = len(env_ids)
        r = torch.empty(n, device=self.device)

        # 1. Sample goal in WORLD frame (relative to robot's current position)
        root_pos = self.robot.data.root_pos_w[env_ids]
        goal_w = torch.zeros(n, 3, device=self.device)
        goal_w[:, 0] = root_pos[:, 0] + r.uniform_(*self.cfg.ranges.pos_x)
        goal_w[:, 1] = root_pos[:, 1] + r.uniform_(*self.cfg.ranges.pos_y)
        goal_w[:, 2] = root_pos[:, 2] + r.uniform_(*self.cfg.ranges.pos_z)
        self._goal_pos_w[env_ids] = goal_w

        # 2. Decide: 4-legged walk or 3-legged manipulation?
        is_manip = torch.rand(n, device=self.device) < self.cfg.rel_manip_envs

        # 3. Proximity check: which front leg hip is closer to goal?
        fr_hip_w = self.robot.data.body_pos_w[env_ids, self._fr_hip_body_id, :]
        fl_hip_w = self.robot.data.body_pos_w[env_ids, self._fl_hip_body_id, :]
        dist_fr = torch.norm(goal_w - fr_hip_w, dim=-1)
        dist_fl = torch.norm(goal_w - fl_hip_w, dim=-1)
        fr_is_closer = dist_fr <= dist_fl

        # 4. Set leg_state — order matches ACTUAL robot joint order (resolved at init)
        leg_state = torch.ones(n, 4, device=self.device)
        fr_selected = is_manip & fr_is_closer
        fl_selected = is_manip & ~fr_is_closer
        zero = torch.tensor(0.0, device=self.device)
        leg_state[:, self._fr_leg_idx] = torch.where(fr_selected, zero, leg_state[:, self._fr_leg_idx])
        leg_state[:, self._fl_leg_idx] = torch.where(fl_selected, zero, leg_state[:, self._fl_leg_idx])
        self._command[env_ids, 6:] = leg_state

    def _update_command(self):
        """Every step: transform world goal to body frame + reachability projection."""
        root_pos = self.robot.data.root_pos_w      # (N, 3)
        root_quat = self.robot.data.root_quat_w    # (N, 4)

        # Transform goal from world to body frame → G_d
        goal_rel_w = self._goal_pos_w - root_pos
        G_d = quat_apply_inverse(root_quat, goal_rel_w)  # (N, 3)
        self._command[:, 0:3] = G_d

        # Reachability map: project G_d onto sphere around selected hip
        fr_hip_w = self.robot.data.body_pos_w[:, self._fr_hip_body_id, :]
        fl_hip_w = self.robot.data.body_pos_w[:, self._fl_hip_body_id, :]
        fr_hip_b = quat_apply_inverse(root_quat, fr_hip_w - root_pos)
        fl_hip_b = quat_apply_inverse(root_quat, fl_hip_w - root_pos)

        # Select hip based on leg_state (using resolved indices)
        leg_state = self._command[:, 6:]
        is_fr_manip = (leg_state[:, self._fr_leg_idx] == 0).unsqueeze(1)
        hip_b = torch.where(is_fr_manip.expand(-1, 3), fr_hip_b, fl_hip_b)

        # Project onto reachability sphere
        goal_rel_hip = G_d - hip_b
        goal_dist = torch.norm(goal_rel_hip, dim=-1, keepdim=True).clamp(min=1e-6)
        clamped_dist = torch.clamp(goal_dist, max=self.cfg.reachability_radius)
        G_B = hip_b + goal_rel_hip / goal_dist * clamped_dist
        self._command[:, 3:6] = G_B

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class GoalPositionCommandCfg(CommandTermCfg):
    """Configuration for ALaM goal position command generator."""

    class_type: type = GoalPositionCommand

    asset_name: str = MISSING
    """Name of the robot asset."""

    ee_body_name: str = ".*diy_link4"
    """Regex pattern for end-effector body names (for metrics)."""

    reachability_radius: float = 0.55
    """Maximum practical reach from hip (meters).
    Go2 standard: thigh(0.21) + calf(0.21) ≈ 0.42m → use 0.34.
    DIY leg: thigh(0.21) + diy_arm(0.48) ≈ 0.69m → use 0.55.
    """

    rel_manip_envs: float = 0.3
    """Fraction of environments doing 3-legged manipulation.
    Ramps to 1.0 via curriculum.
    """

    @configclass
    class Ranges:
        """Goal sampling ranges in WORLD frame (relative to robot position, meters).
        Start with reachable range for stability. Expand via curriculum later.
        Paper evaluation uses -5m to 5m.
        """

        pos_x: tuple[float, float] = (0.1, 0.6)
        """Range for x offset from robot (forward)."""

        pos_y: tuple[float, float] = (-0.3, 0.3)
        """Range for y offset from robot (lateral)."""

        pos_z: tuple[float, float] = (-0.2, 0.1)
        """Range for z offset from robot (height)."""

    ranges: Ranges = Ranges()
