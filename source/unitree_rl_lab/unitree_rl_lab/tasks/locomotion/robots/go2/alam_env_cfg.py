"""ALaM pedipulation on standard Go2 — faithful to paper.

Changes from vanilla Go2 locomotion:
- Goals in WORLD frame, transformed to body frame each step
- Reachability map projection each step
- Proximity-based leg selection
- Dual goal tracking (raw G_d + projected G_B)
- Foot contact indicator & foot positions in observations
- Privileged encoder observations (CoF velocity, CoM pos)
- Manipulation leg uses relative (incremental) position control
- Velocity commands kept for locomotion learning (small range)
"""

from isaaclab.envs.mdp.observations import generated_commands
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands.goal_position_command import GoalPositionCommandCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
)

_FEET_BODIES = [".*_foot"]
_FEET_BODIES_CFG = SceneEntityCfg("robot", body_names=_FEET_BODIES)
_EE_BODIES_CFG = SceneEntityCfg("robot", body_names=["F._foot"])

# ---------------------------------------------------------------------------
# ALaM configs for standard Go2
# ---------------------------------------------------------------------------


@configclass
class ALaMCommandsCfg(CommandsCfg):
    """Velocity (small range) + world-frame EE goal."""

    ee_goal = GoalPositionCommandCfg(
        asset_name="robot",
        ee_body_name="F._foot",
        resampling_time_range=(5.0, 10.0),
        debug_vis=False,
        # Go2: thigh(0.213) + calf(0.213) ≈ 0.43m, use 80%
        reachability_radius=0.34,
        rel_manip_envs=0.3,
        ranges=GoalPositionCommandCfg.Ranges(
            pos_x=(-2.0, 2.0),
            pos_y=(-2.0, 2.0),
            pos_z=(-0.2, 0.1),
        ),
    )


@configclass
class ALaMObservationsCfg(ObservationsCfg):
    """Policy + command + privileged observation groups."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Paper Table 1: foot contact indicator b ∈ R^4
        foot_contact = ObsTerm(
            func=mdp.foot_contact_indicator,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_BODIES)},
        )
        # Paper Table 1: foot position in base frame r_a
        foot_pos = ObsTerm(
            func=mdp.foot_pos_base,
            params={"asset_cfg": _FEET_BODIES_CFG},
        )

    policy: PolicyCfg = PolicyCfg()

    # Command group: [G_d(3), G_B(3), leg_state(4)] = 10 dims
    @configclass
    class CommandCfg(ObsGroup):
        ee_goal_commands = ObsTerm(
            func=generated_commands, clip=(-100, 100),
            params={"command_name": "ee_goal"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()

    # Privileged group (training only)
    @configclass
    class PrivilegedCfg(ObsGroup):
        cof_velocity = ObsTerm(
            func=mdp.cof_velocity,
            params={"feet_cfg": _FEET_BODIES_CFG},
        )
        com_pos_base = ObsTerm(func=mdp.com_pos_base)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class ALaMRewardsCfg(RewardsCfg):
    """Paper Tables 2 & 3 rewards."""

    # ---- Balance (Table 2) ----
    gravitational_moment = RewTerm(
        func=mdp.gravitational_moment_reward, weight=1.0,
        params={"feet_cfg": _FEET_BODIES_CFG, "sigma": 10.0},
    )
    feet_area = RewTerm(
        func=mdp.locomotion_feet_area_reward, weight=0.5,
        params={"feet_cfg": _FEET_BODIES_CFG, "sigma": 0.05},
    )

    # ---- Manipulation (Table 3) ----
    # Projected goal tracking (G_B) — curriculum: 0 → 8.0
    ee_tracking = RewTerm(
        func=mdp.ee_goal_tracking, weight=0.0,
        params={"command_name": "ee_goal", "ee_cfg": _EE_BODIES_CFG, "sigma": 0.1},
    )
    # Raw goal tracking (G_d) — drives robot to walk toward goal
    ee_raw_tracking = RewTerm(
        func=mdp.ee_raw_goal_tracking, weight=0.0,
        params={"command_name": "ee_goal", "ee_cfg": _EE_BODIES_CFG, "sigma": 0.5},
    )


@configclass
class ALaMCurriculumCfg(CurriculumCfg):
    """Curriculum: ramp ee_tracking weight + manipulation env ratio."""

    manip_tracking_levels = CurrTerm(
        func=mdp.manipulation_reward_curriculum,
        params={
            "reward_term_name": "ee_tracking",
            "max_weight": 8.0,
            "step_size": 0.5,
            "upgrade_interval": 2000,
            "stability_term_name": "alive",
            "stability_threshold": 0.3,
            "command_name": "ee_goal",
            "max_manip_ratio": 1.0,
            "manip_ratio_step": 0.1,
        },
    )


@configclass
class ALaMRobotEnvCfg(RobotEnvCfg):
    """ALaM pedipulation training on standard Go2."""

    commands: ALaMCommandsCfg = ALaMCommandsCfg()
    observations: ALaMObservationsCfg = ALaMObservationsCfg()
    rewards: ALaMRewardsCfg = ALaMRewardsCfg()
    curriculum: ALaMCurriculumCfg = ALaMCurriculumCfg()


@configclass
class ALaMRobotPlayEnvCfg(RobotPlayEnvCfg):
    """ALaM pedipulation play/eval on standard Go2."""

    commands: ALaMCommandsCfg = ALaMCommandsCfg()
    observations: ALaMObservationsCfg = ALaMObservationsCfg()
    rewards: ALaMRewardsCfg = ALaMRewardsCfg()
    curriculum: ALaMCurriculumCfg = ALaMCurriculumCfg()
