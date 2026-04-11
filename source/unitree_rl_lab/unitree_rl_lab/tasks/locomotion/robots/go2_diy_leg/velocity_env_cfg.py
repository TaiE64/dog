from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.events import randomize_rigid_body_mass, reset_joints_by_offset
from isaaclab.envs.mdp.rewards import undesired_contacts as undesired_contacts_func
from isaaclab.envs.mdp.terminations import bad_orientation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab.envs.mdp.observations import generated_commands

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_DIY_LEG_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.events import randomize_joint_position_targets
from unitree_rl_lab.tasks.locomotion.mdp.commands.goal_position_command import GoalPositionCommandCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ActionsCfg as Go2ActionsCfg,
    CommandsCfg as Go2CommandsCfg,
    CurriculumCfg as Go2CurriculumCfg,
    EventCfg as Go2EventCfg,
    ObservationsCfg as Go2ObservationsCfg,
    RewardsCfg as Go2RewardsCfg,
    RobotEnvCfg as Go2RobotEnvCfg,
    RobotPlayEnvCfg as Go2RobotPlayEnvCfg,
    RobotSceneCfg as Go2RobotSceneCfg,
    TerminationsCfg as Go2TerminationsCfg,
)


@configclass
class RobotSceneCfg(Go2RobotSceneCfg):
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class EventCfg(Go2EventCfg):
    # reset diy joints to small random offsets (avoid large disturbance at episode start)
    reset_diy_joints = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*diy_joint.*"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # # randomize mass on diy arm end-effectors to simulate grasped objects
    # add_diy_end_effector_mass = EventTerm(
    #     func=randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*diy_link4"]),
    #         "mass_distribution_params": (0.0, 0.5),
    #         "operation": "add",
    #     },
    # )

    # # periodically set random diy joint targets (PD controller moves smoothly)
    # randomize_diy_joints = EventTerm(
    #     func=randomize_joint_position_targets,
    #     mode="interval",
    #     interval_range_s=(3.0, 8.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*diy_joint.*"]),
    #         "position_range": (-0.8, 0.8),
    #     },
    # )


_FEET_BODIES = [".*_foot", ".*diy_base_link"]
_LEG_JOINTS_CFG = SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
_DIY_JOINTS_CFG = SceneEntityCfg("robot", joint_names=[".*diy_joint.*"])

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


@configclass
class ObservationsCfg(Go2ObservationsCfg):
    @configclass
    class PolicyCfg(Go2ObservationsCfg.PolicyCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": _LEG_JOINTS_CFG},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": _LEG_JOINTS_CFG},
        )
        # DIY joint observations
        diy_joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": _DIY_JOINTS_CFG},
        )
        diy_joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": _DIY_JOINTS_CFG},
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(Go2ObservationsCfg.CriticCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100),
            params={"asset_cfg": _LEG_JOINTS_CFG},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100),
            params={"asset_cfg": _LEG_JOINTS_CFG},
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort, scale=0.01, clip=(-100, 100),
            params={"asset_cfg": _LEG_JOINTS_CFG},
        )
        # DIY joint observations for critic
        diy_joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100),
            params={"asset_cfg": _DIY_JOINTS_CFG},
        )
        diy_joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100),
            params={"asset_cfg": _DIY_JOINTS_CFG},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg(Go2RewardsCfg):
    # alive bonus: reward for not falling
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # stronger velocity tracking reward
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5}
    )

    # override: penalize non-foot contacts, but allow diy_base_link (front "calf+foot") to touch ground
    undesired_contacts = RewTerm(
        func=undesired_contacts_func,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Head_.*", ".*_hip", ".*_thigh", "R._calf", ".*diy_link[1-4]"],
            ),
        },
    )

    # override: front legs use diy_base_link as foot instead of *_foot
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_BODIES),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_BODIES)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET_BODIES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_BODIES),
        },
    )

    # override: exclude diy joints from joint-level penalties
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": _LEG_JOINTS_CFG})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": _LEG_JOINTS_CFG})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4, params={"asset_cfg": _LEG_JOINTS_CFG})
    energy = RewTerm(func=mdp.energy, weight=-2e-5, params={"asset_cfg": _LEG_JOINTS_CFG})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": _LEG_JOINTS_CFG})

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )


@configclass
class TerminationsCfg(Go2TerminationsCfg):
    # relax orientation limit: diy arms shift center of mass, need more tolerance
    bad_orientation = DoneTerm(func=bad_orientation, params={"limit_angle": 0.8})


@configclass
class ActionsCfg(Go2ActionsCfg):
    """Locomotion-only actions (diy joints held at default position)."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ALaMActionsCfg(Go2ActionsCfg):
    """ALaM dual-policy actions: locomotion legs + manipulation arms.

    Action order: [12 leg joints, 8 diy joints] = 20 total.
    The ALaMActorCritic network outputs loco actions (first 12) from the
    locomotion actor and manip actions (last 8) from the manipulation actor.
    """

    # Locomotion: 12 leg joints (hip/thigh/calf × 4 legs)
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )
    # Manipulation: diy_joint1 (prismatic, 3cm/s) → velocity control
    DiyJoint1VelocityAction = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*diy_joint1"],
        scale=0.03,  # output ±1 → ±0.03 m/s (full speed)
        use_default_offset=False,
        clip={".*": (-1.0, 1.0)},
    )
    # Manipulation: diy_joint2-4 (revolute) → position control
    DiyJointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*diy_joint2", ".*diy_joint3", ".*diy_joint4"],
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class RobotEnvCfg(Go2RobotEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 20
            self.scene.terrain.terrain_generator.num_cols = 20
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class RobotPlayEnvCfg(Go2RobotPlayEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=32, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


# ---------------------------------------------------------------------------
# ALaM variants: 3-legged locomotion + 1 front leg manipulation
# Reference: "Robust Pedipulation on Quadruped Robots via Gravitational-moment
#             Minimization", Shin et al., IJCAS 2025
# ---------------------------------------------------------------------------

_EE_BODIES_CFG = SceneEntityCfg("robot", body_names=[".*diy_link4"])
_FEET_BODIES_CFG = SceneEntityCfg("robot", body_names=_FEET_BODIES)


@configclass
class ALaMCommandsCfg(Go2CommandsCfg):
    """Velocity + world-frame EE goal (proximity check + reachability map)."""

    ee_goal = GoalPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        debug_vis=False,
        # thigh(0.21) + diy_arm(0.48) ≈ 0.69m, use 80%
        reachability_radius=0.55,
        rel_manip_envs=0.3,
        ranges=GoalPositionCommandCfg.Ranges(
            pos_x=(0.1, 0.6),
            pos_y=(-0.3, 0.3),
            pos_z=(-0.2, 0.1),
        ),
    )


@configclass
class ALaMObservationsCfg(ObservationsCfg):
    """Full ALaM observations: policy + command + privileged (paper Table 1)."""

    # PolicyCfg: inherit base observations, no extra terms for now
    # (foot_contact and foot_pos can be added after training stabilizes)

    # Command: [G_d(3), G_B(3), leg_state(4)] = 10 dims
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

    # Privileged: CoF vel + CoM pos (domain rand can be added later)
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
    """Full ALaM rewards faithful to paper Tables 2 & 3."""

    # ---- Balance (Table 2) ----
    gravitational_moment = RewTerm(
        func=mdp.gravitational_moment_reward, weight=1.0,
        params={"feet_cfg": _FEET_BODIES_CFG, "sigma": 10.0},
    )
    feet_area = RewTerm(
        func=mdp.locomotion_feet_area_reward, weight=0.5,
        params={"feet_cfg": _FEET_BODIES_CFG, "sigma": 0.05},
    )

    # ---- Manipulation task (Table 3) ----
    # Projected goal tracking G_B — curriculum: 0 → 8.0
    ee_tracking = RewTerm(
        func=mdp.ee_goal_tracking, weight=0.0,
        params={"command_name": "ee_goal", "ee_cfg": _EE_BODIES_CFG, "sigma": 0.1},
    )

    # ---- Manipulation regularization ----
    diy_joint_vel = RewTerm(
        func=mdp.diy_joint_vel_penalty, weight=-1e-4,
        params={"asset_cfg": _DIY_JOINTS_CFG},
    )
    diy_joint_acc = RewTerm(
        func=mdp.diy_joint_acc_penalty, weight=-1e-9,
        params={"asset_cfg": _DIY_JOINTS_CFG},
    )
    diy_action_rate = RewTerm(
        func=mdp.diy_action_rate, weight=-0.005,
        params={"loco_action_dim": 12},
    )


@configclass
class ALaMCurriculumCfg(Go2CurriculumCfg):
    """Curriculum: ramp ee_tracking + raw tracking + manip ratio."""

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
    """ALaM training: 3-legged loco + 1 front leg manipulation."""

    actions: ALaMActionsCfg = ALaMActionsCfg()
    commands: ALaMCommandsCfg = ALaMCommandsCfg()
    observations: ALaMObservationsCfg = ALaMObservationsCfg()
    rewards: ALaMRewardsCfg = ALaMRewardsCfg()
    curriculum: ALaMCurriculumCfg = ALaMCurriculumCfg()


@configclass
class ALaMRobotPlayEnvCfg(RobotPlayEnvCfg):
    """ALaM play/eval config."""

    actions: ALaMActionsCfg = ALaMActionsCfg()
    commands: ALaMCommandsCfg = ALaMCommandsCfg()
    observations: ALaMObservationsCfg = ALaMObservationsCfg()
    rewards: ALaMRewardsCfg = ALaMRewardsCfg()
