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

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_DIY_LEG_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.events import randomize_joint_position_targets
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ActionsCfg as Go2ActionsCfg,
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
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
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
