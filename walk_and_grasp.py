"""Vision-driven walk + DIY-arm grasp demo in MuJoCo.

Pipeline:
    head_cam (RGB + depth)
        │
        ▼ HSV threshold for the red target cube → (u, v) + depth(u, v)
        ▼ deproject to camera frame, transform to world frame → goal_world
        │
        ▼
    State machine:
       SEARCH      : no goal yet, slow yaw rotation to scan
       APPROACH    : walk toward goal until base is within 0.55m
       EXTEND      : ramp DIY 1/2/3 from HOLD to REACH_POSE (arm out)
       SERVO       : body cmd_vel drives EE onto cube (arm fixed)
       CLOSE       : joint4 → CLOSE, hold briefly
       LIFT_TEST   : raise arm; if cube_z follows (Δz > LIFT_DETECT_M) →
                     real grasp; otherwise gripper closed on air
       RETRY_BACKUP: back up + side-step, then loop back to EXTEND
       REACQUIRE   : cube fell out of FOV during APPROACH, back up to find it
       DONE        : freeze, grasp_success set

The walk policy runs continuously for the 12 leg joints — all 4 feet stay on
the ground throughout (no 3-leg balance problem). Only the 4 DIY arm joints
are commanded by this script: joint 1/2/3 by IK during REACH, joint4 by
explicit OPEN/CLOSE values.

Usage:
    python walk_and_grasp.py
    python walk_and_grasp.py --cube 0.8 -0.2 0.025      # spawn cube elsewhere
    python walk_and_grasp.py --headless                  # no viewer (smoke test)
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer as viewer

try:
    import onnxruntime as rt
except ImportError:
    raise ImportError("pip install onnxruntime")

# rerun is optional — only loaded if --rerun is passed
_rerun_module = None


def _get_rerun():
    global _rerun_module
    if _rerun_module is None:
        try:
            import rerun as rr

            _rerun_module = rr
        except ImportError:
            raise ImportError("pip install rerun-sdk")
    return _rerun_module


_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_PATH = os.path.join(_HERE, "leg_manipulator", "mjcf", "go2_diyleg.xml")
_ONNX_PATH = os.path.join(_HERE, "walk_model", "go2_diyleg_walk_policy_6200.onnx")

# ---- Walk policy obs/action conventions ----------------------------------
ISAAC_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
DEFAULT_OFFSETS = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.560,
        0.560,
        0.401,
        0.401,
        1.466,
        1.466,
        -1.215,
        -1.215,
    ],
    dtype=np.float32,
)
ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05
ACTION_SCALE = 0.25

# ---- DIY arm conventions ------------------------------------------------
FL_DIY_JOINT_NAMES = [
    "FL_diy_joint1",
    "FL_diy_joint2",
    "FL_diy_joint3",
    "FL_diy_joint4",
]
FR_DIY_JOINT_NAMES = ["diy_joint1", "diy_joint2", "diy_joint3", "diy_joint4"]
# joint1 prismatic slider (range 0..0.105m). Velocity-controlled at J1_SLEW_RATE.
# Non-grasp phases retract; grasp phases extend (capped at J1_GRASP_MAX so IK
# can't leave the slider parked at HOLD and reach with j2/j3 alone — that
# would cramp the arm and make EXTEND look like a "teleport").
J1_RETRACTED = 0.10
J1_SLEW_RATE = 0.03  # m/s
J1_GRASP_MAX = 0.020  # m — IK upper bound on j1 during grasp poses
# Hold pose during walk: joint2=-0.3 lifts arm tip above horizontal, joint3=0.
FL_DIY_HOLD_123 = np.array([J1_RETRACTED, -0.3, 0.0], dtype=np.float32)
J4_OPEN = 2.0  # gripper open  (joint4 range [0.2, 2.3])
J4_CLOSE = 0.5  # gripper closed/grasping

# ---- Reactive controller params -----------------------------------------
# No hard-coded "stop distance" or object size. The decisions are:
#   - WHEN TO STOP  ← IK feasibility test on the perceived goal
#   - WHERE TO REACH ← goal = object center (estimated from bbox + depth)
# This generalizes to objects of arbitrary size/shape because the geometry
# decisions come from sensing, not from constants.

# Vision-loss handling
VISION_LOST_TIMEOUT = 1.5  # s — APPROACH watchdog (cube in field-of-view loss)
SEARCH_YAW_RATE = 0.4  # rad/s — initial-search rotation
REACQUIRE_BACK_VX = -0.25  # m/s — backward speed when reacquiring (cube was
# cached but is currently out of FOV → most likely
# we're just too close, backing up brings it back
# into view)
REACQUIRE_TIMEOUT_S = 4.0  # s — if can't reacquire by then, fall through to
# SEARCH (rotate scan)

EXTEND_RAMP_S = 1.0  # s — smooth blend HOLD→REACH_POSE
JOINT_SLEW_RATE = 0.8  # rad/s — cap how fast the SERVO IK output can
# change. Combined with warm_only=True it
# prevents arm "teleport" between IK branches
# when the body wobbles or residuals tie.
# Bar long axis in world (current MJCF: dumbbell handle along world Y).
# Used by PRE_ALIGN to position the body perpendicular to the bar so the
# gripper's closing direction lines up with the bar's narrow axis. Could be
# derived per-frame from vision (bar bbox direction → world via cam_R) for
# generality; hardcoded for now to match the demo MJCF.
BAR_AXIS_XY = np.array([0.0, 1.0])
APPROACH_PERP_DIST = 0.40  # m — clearance for FL leg (at base+0.193,+0.046m,
# ~5cm radius) to not bump basket (half-width
# 0.075m). Min safe: 0.193+0.05+0.075 = 0.318m.
# 0.40 gives 8cm margin. Arm reach is ~0.29m so
# SERVO body floor 0.32m gets EE close to goal.
PRE_ALIGN_YAW_TOL = 0.15  # rad ≈ 8.6° — close enough to perpendicular
# yaw. Walk policy can't yaw precisely when
# standing still (no forward motion to anchor),
# so 0.08 (≈5°) was unreachable in practice.
PRE_ALIGN_XY_TOL = 0.08  # m — close enough to approach_xy
PRE_ALIGN_TIMEOUT = 8.0  # s — give up and try EXTEND anyway

SERVO_TOL_M = 0.02  # m — EE→cube ≤ this counts as "covered" → CLOSE.
# Reduced to 0.02 so the IK pivot gets close enough for the 4cm jaws to straddle the bar.
GRIPPER_OFFSET_BACK = 0.0   # m — Disabled offset, rely on SERVO_TOL_M to leave room for jaws
GRIPPER_OFFSET_UP = 0.0     # m — Disabled offset
SERVO_KP = 1.5  # m/s of body cmd per m of EE→cube error
SERVO_MIN_VX = 0.20  # m/s — walk policy has a deadband below this;
# we floor forward cmd here so the dog actually
# walks while IK residual is still large.
SERVO_MAX_VX = 0.30  # m/s body forward cap during SERVO
SERVO_MAX_WZ = 0.5  # rad/s yaw cap
SERVO_TIMEOUT_S = 6.0  # s — if can't get under tolerance in this long, close anyway
CLOSE_HOLD_S = 0.6
# Active lift test: after closing, raise the arm by LIFT_HEIGHT_M and watch
# the cube. If the cube actually came up with the arm by ≥ LIFT_DETECT_M,
# we truly grabbed it; if it stayed put, the gripper closed on air.
LIFT_HEIGHT_M = 0.06  # m — raise arm this much above grasp z
LIFT_RAMP_S = 0.6  # s — time to ramp arm to lifted pose
LIFT_HOLD_S = 0.6  # s — hold lifted, then read cube z
LIFT_DETECT_M = 0.020  # m — cube must rise at least 2cm during the
# 6cm arm raise to count as a real grasp.
RETRY_BACKUP_S = 1.0  # s — back up between retries (longer = more
# visible "trying again", and base is reset to
# a different pose for the next SERVO attempt)
RETRY_SIDE_STEP = 0.10  # m/s lateral wobble during RETRY_BACKUP — gives
# next attempt a slightly different vantage
RETRY_MAX = 99999  # after this many failed LIFT_TESTs, give up
# (DONE with grasp_success=False) instead of
# looping forever
GRASP_HOLD_RADIUS = 0.13  # m — gripper-jaw extent + slack. Without
# explicit jaw geoms in the MJCF, joint4 can't
# *actually* close on the cube; we instead
# declare success when EE is within this radius.

# REACH_POSE_FORWARD/SIDE: the EE landing target in body frame is at
# (forward, side, target_z) where target_z is the OBJECT's height. Reach
# pose is recomputed via IK each time we EXTEND so the arm naturally
# adapts to floor / table / shelf heights.
REACH_POSE_FORWARD = 0.38
REACH_POSE_SIDE = 0.08  # FL arm's natural left bias. Compensated NOT by
# yawing the dog (which made body face handle
# diagonally) but by shifting approach_xy by
# -SIDE along the bar's axis — see
# _compute_approach_pose.

WALK_SPEED_MAX = 0.5  # m/s


# =========================================================================
# Vision: render head_cam, find red object, target its narrowest section
# (so the gripper has a chance to close on it).
# =========================================================================
# Approx jaw opening of FL_diy_link4 gripper (~3-4cm). Thin sections below
# this width are considered graspable.
GRIPPER_OPEN_M = 0.04
# We track FL_diy_link4's body origin (the joint4 axis) as the "EE point".
# The actual pinch point sits a few cm forward, but with j4 at J4_CLOSE the
# jaws straddle the body origin closely enough that landing it on the bar
# yields a successful grasp.


class Vision:
    def __init__(
        self,
        model: mujoco.MjModel,
        cam_name: str = "head_cam",
        height: int = 240,
        width: int = 320,
    ):
        self.model = model
        self.height, self.width = height, width
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera {cam_name!r} not found")
        # Pinhole intrinsics derived from FOV
        fovy_deg = float(model.cam_fovy[self.cam_id])
        self.fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg / 2.0))
        self.fx = self.fy  # square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0
        # Lazily allocate renderer (it's expensive)
        self._renderer = None

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)

    def _deproject(self, u: float, v: float, z: float, cam_pos, cam_R) -> np.ndarray:
        """Pixel + depth → world-frame 3D point."""
        x_cam = (u - self.cx) / self.fx * z
        y_cam = -(v - self.cy) / self.fy * z
        z_cam = -z
        p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
        return cam_R @ p_cam + cam_pos

    def detect(self, data: mujoco.MjData):
        """Detect target's NARROWEST graspable section in 3D world frame.

        Pipeline:
          1. HSV mask of red pixels
          2. Compute per-column and per-row "thickness" of the mask. The
             dimension along which the object is LONGER is its long axis;
             the perpendicular dimension is the "narrow" one.
          3. Walk along the long axis and find the column/row with the
             FEWEST red pixels (the object's thinnest cross-section, e.g.
             dumbbell handle, bottle neck, banana waist).
          4. Centroid of that thin slice → grasp pixel.
          5. Deproject + camera→world to get 3D goal.

        Returns dict + rgb (always rgb for viz).
        """
        self._ensure_renderer()
        self._renderer.update_scene(data, camera=self.cam_id)
        rgb = self._renderer.render().copy()

        red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
        if int(red.sum()) < 30:
            return None, rgb

        ys, xs = np.where(red)
        u_min, u_max = int(xs.min()), int(xs.max())
        v_min, v_max = int(ys.min()), int(ys.max())
        bbox_w = u_max - u_min + 1
        bbox_h = v_max - v_min + 1
        u_c, v_c = float(xs.mean()), float(ys.mean())

        # Find the thin slice + its two silhouette edges (the diametrically-
        # opposite surface points across the bar's narrow direction):
        #   - horizontal bar (long u-axis): two pixels at same u — top + bottom rows
        #   - vertical bar   (long v-axis): two pixels at same v — left + right cols
        # Deprojecting each edge and averaging in WORLD frame gives the bar's
        # central axis exactly: any pair of diametrically-opposite surface
        # points averages to the center, regardless of camera angle.
        edge1_pixel, edge2_pixel = None, None

        def _longest_thin_run(thickness):
            """Find the longest CONTIGUOUS run of columns/rows whose
            thickness is in the bottom half of the (positive) range. The
            bar/handle is the long thin region between the plates; plate
            edges yield only 1-2 isolated thin columns that get filtered.
            Returns (start_idx, end_idx) inclusive in the local bbox frame.
            """
            mask = thickness > 0
            if not mask.any():
                return None
            # threshold = midpoint between min nonzero and max
            tmin = int(thickness[mask].min())
            tmax = int(thickness[mask].max())
            thresh = max(tmin, (tmin + tmax) // 2 - 1)
            thin_mask = mask & (thickness <= thresh)
            # Walk to find longest contiguous run
            best_len, best_lo, best_hi = 0, None, None
            cur_lo = None
            for i, t in enumerate(thin_mask):
                if t:
                    if cur_lo is None:
                        cur_lo = i
                    cur_hi = i
                    if cur_hi - cur_lo + 1 > best_len:
                        best_len = cur_hi - cur_lo + 1
                        best_lo, best_hi = cur_lo, cur_hi
                else:
                    cur_lo = None
            return (best_lo, best_hi) if best_lo is not None else None

        if bbox_w >= bbox_h:
            col_thickness = red[v_min : v_max + 1, u_min : u_max + 1].sum(axis=0)
            run = _longest_thin_run(col_thickness)
            if run is not None:
                lo, hi = run
                u_thin_local = (lo + hi) // 2
                u_thin = u_min + u_thin_local
                rows_red = np.where(red[:, u_thin])[0]
                if rows_red.size:
                    v_top = int(rows_red.min())
                    v_bot = int(rows_red.max())
                    edge1_pixel = (u_thin, v_top)
                    edge2_pixel = (u_thin, v_bot)
                    v_thin = (v_top + v_bot) / 2.0
                else:
                    v_thin = v_c
                thin_px = int(col_thickness[u_thin_local])
                u_c, v_c = float(u_thin), v_thin
            else:
                thin_px = -1
        else:
            row_thickness = red[v_min : v_max + 1, u_min : u_max + 1].sum(axis=1)
            run = _longest_thin_run(row_thickness)
            if run is not None:
                lo, hi = run
                v_thin_local = (lo + hi) // 2
                v_thin = v_min + v_thin_local
                cols_red = np.where(red[v_thin, :])[0]
                if cols_red.size:
                    u_left = int(cols_red.min())
                    u_right = int(cols_red.max())
                    edge1_pixel = (u_left, v_thin)
                    edge2_pixel = (u_right, v_thin)
                    u_thin = (u_left + u_right) / 2.0
                else:
                    u_thin = u_c
                thin_px = int(row_thickness[v_thin_local])
                u_c, v_c = u_thin, float(v_thin)
            else:
                thin_px = -1

        # Depth pass — read depth at the two silhouette edges. Their world-
        # coordinate midpoint = bar central axis.
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self.cam_id)
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        if edge1_pixel is None or edge2_pixel is None:
            edge1_pixel = (int(round(u_c)), int(round(v_c)))
            edge2_pixel = edge1_pixel
        u1, v1 = edge1_pixel
        u2, v2 = edge2_pixel
        z1 = float(depth[v1, u1])
        z2 = float(depth[v2, u2])
        if z1 <= 0.0 or z1 > 5.0:
            return None, rgb
        if z2 <= 0.0 or z2 > 5.0:
            z2 = z1

        cam_pos = data.cam_xpos[self.cam_id].copy()
        cam_R = data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        edge1_world = self._deproject(u1, v1, z1, cam_pos, cam_R)
        edge2_world = self._deproject(u2, v2, z2, cam_pos, cam_R)
        goal_world = ((edge1_world + edge2_world) / 2.0).astype(np.float64)

        # Thin-section width in meters: 1 pixel at depth z spans z/fx meters.
        thin_m = thin_px * (z1 / self.fx) if thin_px > 0 else float("nan")

        # Overlay: draw bbox + mark grasp point in green
        if rgb.flags.writeable:
            rgb[v_min, u_min : u_max + 1] = [0, 255, 0]
            rgb[v_max, u_min : u_max + 1] = [0, 255, 0]
            rgb[v_min : v_max + 1, u_min] = [0, 255, 0]
            rgb[v_min : v_max + 1, u_max] = [0, 255, 0]
            ui = int(np.clip(round(u_c), 2, self.width - 3))
            vi = int(np.clip(round(v_c), 2, self.height - 3))
            rgb[vi - 2 : vi + 3, ui - 2 : ui + 3] = [0, 255, 0]

        # Partial-view guard: if the red bbox touches any image edge, we're
        # only seeing part of the object → grasp point on the visible chunk
        # is wrong (e.g. picks one plate instead of the handle middle).
        # Caller should treat partial detections as "no fresh info".
        edge_margin = 1
        partial = (
            u_min <= edge_margin
            or u_max >= self.width - 1 - edge_margin
            or v_min <= edge_margin
            or v_max >= self.height - 1 - edge_margin
        )

        return {
            "goal_world": goal_world.astype(np.float32),
            "thin_width_m": thin_m,
            "thin_pixels": thin_px,
            "graspable": bool(thin_m < GRIPPER_OPEN_M) if thin_m == thin_m else False,
            "partial": partial,
        }, rgb


# =========================================================================
# 3-DoF IK on DIY arm (joint 1, 2, 3 — joint 4 is gripper, not part of IK)
# =========================================================================
class ArmIK3:
    """Solve DIY joint1/2/3 to put FL_diy_link4 at a world target. Damped LS."""

    def __init__(self, model: mujoco.MjModel, side: str = "FL", ee_local_offset=None):
        self.model = model
        self.ee_local_offset = np.array(ee_local_offset) if ee_local_offset is not None else None
        ee_body_name = f"{side}_diy_link4" if side == "FL" else "diy_link4"
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        if self.body_id < 0:
            raise ValueError(f"Body {ee_body_name} not found")

        diy_names = (
            [f"{side}_diy_joint{i}" for i in (1, 2, 3)]
            if side == "FL"
            else [f"diy_joint{i}" for i in (1, 2, 3)]
        )
        self._jids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in diy_names]
        )
        if (self._jids < 0).any():
            raise ValueError("Missing DIY joints 1/2/3")
        self._qadrs = np.array([model.jnt_qposadr[j] for j in self._jids])
        self._dofadrs = np.array([model.jnt_dofadr[j] for j in self._jids])
        self._ranges = np.array([model.jnt_range[j] for j in self._jids])

        self._wdata = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))

    # Multi-start seeds: cover the joint2/3 workspace coarsely so DLS doesn't
    # get trapped in a local minimum near the cube. Each entry is the seed
    # value for [j1, j2, j3]; the first seed = current arm pose (warm start).
    _IK_SEEDS = [
        None,  # warm start (use current data.qpos)
        (0.0, -0.3, -0.6),  # forward / down — typical reach
        (0.0, -0.6, -1.0),  # more extended
        (0.05, -0.4, -0.8),  # joint1 partially extended
        (0.0, 0.3, 0.6),  # mirror seed (avoid sign-trap)
    ]

    def _solve_once(self, target, max_iter, damp, tol, step_clip, j1_max=None):
        last_err = np.inf
        for _ in range(max_iter):
            mujoco.mj_kinematics(self.model, self._wdata)
            mujoco.mj_comPos(self.model, self._wdata)
            tip = self._wdata.body(self.body_id).xpos.copy()
            if self.ee_local_offset is not None:
                mat = self._wdata.body(self.body_id).xmat.reshape(3, 3)
                tip += mat @ self.ee_local_offset
            err = target - tip
            mag = float(np.linalg.norm(err))
            last_err = mag
            if mag < tol:
                break
            mujoco.mj_jac(self.model, self._wdata, self._jacp, None, tip, self.body_id)
            J = self._jacp[:, self._dofadrs]
            JJT = J @ J.T + (damp**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            dq = np.clip(dq, -step_clip, step_clip)
            self._wdata.qpos[self._qadrs] += dq
            for k in range(3):
                lo, hi = self._ranges[k]
                # Cap joint1 ≤ j1_max so IK is forced to actually extend the
                # slider (low value) rather than leaving it retracted (high
                # value) and reaching only with j2/j3 rotation.
                if k == 0 and j1_max is not None:
                    hi = min(hi, j1_max)
                self._wdata.qpos[self._qadrs[k]] = float(
                    np.clip(self._wdata.qpos[self._qadrs[k]], lo, hi)
                )
        return self._wdata.qpos[self._qadrs].copy().astype(np.float32), last_err

    def solve(
        self,
        data,
        target_world,
        max_iter=60,
        damp=0.05,
        tol=5e-3,
        step_clip=0.10,
        warm_only=False,
        j1_max=None,
    ):
        # warm_only=True: skip multi-seed and just iterate from the current
        # qpos. Use this in SERVO so the IK solution is CONTINUOUS frame to
        # frame — multi-seed picks the lowest-residual branch each tick,
        # which can flip between IK branches and teleport the arm.
        target = np.asarray(target_world, dtype=np.float64)
        base_qpos = data.qpos.copy()
        best_q, best_err = None, np.inf
        seeds = [None] if warm_only else self._IK_SEEDS
        for seed in seeds:
            self._wdata.qpos[:] = base_qpos
            self._wdata.qvel[:] = 0.0
            if seed is not None:
                for k, v in enumerate(seed):
                    lo, hi = self._ranges[k]
                    if k == 0 and j1_max is not None:
                        hi = min(hi, j1_max)
                    self._wdata.qpos[self._qadrs[k]] = float(np.clip(v, lo, hi))
            elif j1_max is not None:
                # warm-start: clamp current j1 down to j1_max so the solver
                # starts from an extended pose instead of staying retracted.
                lo, hi = self._ranges[0]
                self._wdata.qpos[self._qadrs[0]] = float(
                    np.clip(self._wdata.qpos[self._qadrs[0]], lo, min(hi, j1_max))
                )
            q, err = self._solve_once(
                target, max_iter, damp, tol, step_clip, j1_max=j1_max
            )
            if err < best_err:
                best_q, best_err = q, err
                if err < tol:
                    break
        return best_q, best_err


# =========================================================================
# Main controller
# =========================================================================
class GraspController:
    def __init__(self, model: mujoco.MjModel, policy_path: str):
        self.policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )
        self._output_names = ["continuous_actions"]

        self._qpos_indices = np.array(
            [model.joint(n).qposadr.item() for n in ISAAC_JOINT_NAMES]
        )
        self._qvel_indices = np.array(
            [model.joint(n).dofadr.item() for n in ISAAC_JOINT_NAMES]
        )
        self._actuator_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_actuator")
                for n in ISAAC_JOINT_NAMES
            ]
        )
        self._fl_diy_act_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_actuator")
                for n in FL_DIY_JOINT_NAMES
            ]
        )
        self._fr_diy_act_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_actuator")
                for n in FR_DIY_JOINT_NAMES
            ]
        )
        self._imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self._fl_ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "FL_diy_link4"
        )
        # Cube body — used for grasp verification only (we don't "cheat" by
        # using its ground truth elsewhere; only after CLOSE to confirm the
        # gripper actually has it).
        # `target_cube` body is actually the shopping basket; the red graspable
        # part is the `target_handle` cylinder geom (offset +0.068m in body z).
        # Track the geom for grasp-verification distance/lift checks; keep the
        # body id for contact filtering (contacts report bodyid, not geomid).
        self._cube_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "target_cube"
        )
        self._handle_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_handle"
        )
        if self._handle_geom_id < 0:
            raise ValueError("geom 'target_handle' not found")
        # All FL DIY arm bodies — for contact-based grasp verification.
        self._fl_arm_body_ids = set()
        for nm in [
            "FL_diy_link1",
            "FL_diy_link2",
            "FL_diy_link3",
            "FL_diy_link4",
            "FL_diy_base_link",
            "FL_diy_foot",
        ]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid >= 0:
                self._fl_arm_body_ids.add(bid)

        self._last_action = np.zeros(12, dtype=np.float32)
        self._n_substeps = int(round(0.02 / model.opt.timestep))
        self._counter = self._n_substeps - 1
        self._step_dt = self._n_substeps * model.opt.timestep

        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.vision = Vision(model)
        # Offset the EE tip 3.5cm forward from the pivot to reach the actual pinch point between the jaws
        self.ee_local_offset = np.array([0.035, 0.0, 0.0], dtype=np.float32)
        self.ik = ArmIK3(model, side="FL", ee_local_offset=self.ee_local_offset)
        self._model = model

        # Per-arm DIY default hold (joint 1/2/3 + joint4 OPEN at start)
        self._fr_diy_hold = np.array([0.05, -1.06, 0.0, J4_OPEN], dtype=np.float32)

        # Reach pose is computed lazily once we know the target height.
        # Default to floor height; gets recomputed when EXTEND is entered
        # so the arm adapts to floor / table / shelf-height objects.
        self._reach_pose, self._ee_offset_body, self._reach_target_world = (
            self._compute_reach_pose(model, target_world_z=0.05)
        )
        print(f"REACH_POSE (default, floor) = {self._reach_pose.round(3)}")
        print(f"EE offset in body frame    = {self._ee_offset_body.round(3)}m")

        # Vision cache
        self.goal_world: np.ndarray | None = None
        self._last_vision_t = -1.0
        self._last_detect_t = -1.0
        self.last_rgb = None
        # Goal locking: collect a few "good" detections (full view) at the
        # start, average them, then lock — close-range detections see only
        # part of the object so the per-frame target jitters; once locked
        # we trust the cached 3D goal until the object actually moves.
        self._goal_lock_buf = (
            []
        )  # list of detected goal_world positions while not locked
        self._goal_locked = False  # once True, stop updating goal_world
        self.GOAL_LOCK_SAMPLES = 5  # accumulate this many good detections
        self.GOAL_MOVED_THRESHOLD = (
            0.15  # m — if new detection is this far from locked, cube moved → unlock
        )

        # State machine: see top-of-file docstring for phase semantics.
        self.phase = "SEARCH"
        self.phase_t = 0.0
        # SERVO progress
        self._servo_best_d = float("inf")
        self._last_ik_err = float("nan")
        # PRE_ALIGN scratch — overwritten each APPROACH/PRE_ALIGN tick.
        self._approach_xy = np.zeros(2, dtype=np.float32)
        self._approach_yaw = 0.0
        # joint1 prismatic slider — velocity-controlled (3cm/s slew).
        self._j1_cmd = float(J1_RETRACTED)
        # Previous SERVO IK output, for joint-space slew limiting.
        self._prev_reach_pose = self._reach_pose.copy()
        # EXTEND task-space interpolation: capture EE world pos at phase entry,
        # interpolate linearly toward _reach_target_world, IK each tick.
        self._extend_ee_start: np.ndarray | None = None
        # Last arm target written to actuators; used to slew-limit j2/j3 across
        # phase transitions (e.g. LIFT_TEST → RETRY_BACKUP) so the arm doesn't
        # teleport from extended pose back to HOLD in a single tick.
        self._last_arm_target_j23 = FL_DIY_HOLD_123[1:].astype(np.float32).copy()
        # Lift-test verification state
        self._lift_cube_z_start = 0.0
        self._lift_reach_pose = self._reach_pose.copy()
        self._lift_from_pose = self._reach_pose.copy()
        self._lift_contact_seen = False
        self.grasp_success = None
        self._retry_count = 0

        print(
            f"GraspController: substeps/policy_step={self._n_substeps}, policy_dt={self._step_dt*1000:.1f}ms"
        )

    def _compute_reach_pose(
        self,
        model: mujoco.MjModel,
        target_world_z: float,
        live_data: mujoco.MjData | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve IK for the arm to reach a target at body-frame
        (REACH_POSE_FORWARD, REACH_POSE_SIDE, target_world_z).
        If live_data is provided, IK is solved at the LIVE base pose so the
        result accounts for body sag during walking (keyframe sits at z=0.40
        but actual standing is ~0.36).
        Returns (reach_pose[3], ee_offset_body[3]).
        """
        wd = mujoco.MjData(model)
        if live_data is not None:
            wd.qpos[:] = live_data.qpos
        else:
            mujoco.mj_resetDataKeyframe(model, wd, 0)
        wd.qvel[:] = 0.0
        mujoco.mj_forward(model, wd)
        base_pos = wd.qpos[0:3].copy()
        yaw = self._quat_yaw(wd.qpos[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        # World target: forward / side relative to base (body x/y), absolute z.
        target_world = np.array(
            [
                base_pos[0] + c * REACH_POSE_FORWARD - s * REACH_POSE_SIDE,
                base_pos[1] + s * REACH_POSE_FORWARD + c * REACH_POSE_SIDE,
                target_world_z,
            ]
        )
        reach_pose, err = self.ik.solve(
            wd, target_world, max_iter=200, damp=0.03, j1_max=J1_GRASP_MAX
        )
        for k, qadr in enumerate(self.ik._qadrs):
            wd.qpos[qadr] = reach_pose[k]
        mujoco.mj_forward(model, wd)
        ee_world = self._ee_tip_world(wd)
        rel = ee_world - base_pos
        ee_body = np.array(
            [
                c * rel[0] + s * rel[1],
                -s * rel[0] + c * rel[1],
                rel[2],
            ]
        )
        if err > 0.06:
            print(
                f"WARN: reach-pose IK residual {err:.3f}m at target_z={target_world_z:.3f} (unreachable?)"
            )
        return (
            reach_pose.astype(np.float32),
            ee_body.astype(np.float32),
            target_world.astype(np.float32),
        )

    # --- helpers ----
    def _compute_approach_pose(self, goal_world, base_pos):
        """Pose to settle into BEFORE final reach. Body forward is
        perpendicular to the bar's long axis (so gripper closes across the
        bar's narrow direction, not along its length), and body sits at
        APPROACH_PERP_DIST on whichever side of the bar is closer to the
        robot's current position. Yaw includes the arm-bias correction so
        the FL arm's natural left-offset aligns with the bar.

        Returns (approach_xy, target_yaw, perp_dir).
        """
        bar_axis = BAR_AXIS_XY / max(float(np.linalg.norm(BAR_AXIS_XY)), 1e-9)
        # Two perpendicular directions in xy plane (unit vectors)
        perp_a = np.array([-bar_axis[1], bar_axis[0]])
        perp_b = -perp_a
        to_robot = base_pos[:2] - goal_world[:2]
        # Pick the perp pointing from cube TOWARD robot — that's the side
        # we'll approach from (closest, no need to go around the bar).
        perp = (
            perp_a
            if float(np.dot(to_robot, perp_a)) > float(np.dot(to_robot, perp_b))
            else perp_b
        )
        # Body forward = -perp (dog faces directly at handle, no yaw tilt).
        forward_yaw = float(np.arctan2(-perp[1], -perp[0]))
        target_yaw = forward_yaw
        # Shift body so the FL arm's natural y-bias lines up with the handle.
        # The shift must be along the robot's local left direction in world frame,
        # which is perpendicular to its forward vector (-perp).
        left_vector = np.array([perp[1], -perp[0]])
        side_shift = left_vector * (-float(self._ee_offset_body[1]))
        approach_xy = goal_world[:2] + perp * APPROACH_PERP_DIST + side_shift
        return (
            approach_xy.astype(np.float32),
            float(target_yaw),
            perp.astype(np.float32),
        )

    def _ee_tip_world(self, data):
        """World position of the true pinch point between the jaws."""
        pivot = data.body(self._fl_ee_id).xpos.copy()
        if self.ee_local_offset is not None:
            mat = data.body(self._fl_ee_id).xmat.reshape(3, 3)
            return pivot + mat @ self.ee_local_offset
        return pivot
    def _slew_reach_pose(self, ik_q):
        """Cap per-tick joint-command change to JOINT_SLEW_RATE * dt.
        IK can flip between solution branches when residuals tie or the
        body wobbles; this turns the would-be teleport into a smooth ramp."""
        max_step = JOINT_SLEW_RATE * self._step_dt
        delta = np.asarray(ik_q, dtype=np.float32) - self._prev_reach_pose
        delta = np.clip(delta, -max_step, max_step)
        slewed = self._prev_reach_pose + delta
        self._prev_reach_pose = slewed.copy()
        return slewed

    @staticmethod
    def _quat_yaw(q):
        w, x, y, z = q
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _cube_in_gripper_contact(self, model, data) -> bool:
        """Honest grasp check: is there an actual MuJoCo contact between the
        cube and any FL-arm body? Position/distance is NOT enough — gripper
        can close on empty air right next to the cube."""
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if (b1 == self._cube_id and b2 in self._fl_arm_body_ids) or (
                b2 == self._cube_id and b1 in self._fl_arm_body_ids
            ):
                return True
        return False

    def _set_servo_command(self, base_pos, base_yaw, cube_world):
        """Body+arm COORDINATION: drive body to the (x, y, yaw) pose where
        the arm's reach-pose tip naturally lands on cube.

        Geometry:
          arm in reach pose has tip at body-local _ee_offset_body.
          for tip world = cube, body must be at:
              ideal_body_xy = cube_xy - R(yaw) @ ee_offset_xy
          and yaw = atan2(cube - body) - ee_lat_angle so that the arm's
          natural left-bias points at cube.

        We don't pick a 1D 'stand distance'. We drive body straight at
        ideal_body. The arm IK in parallel takes up whatever slack remains
        after the body's imprecision.

        Lateral cmd_vel stays 0 — quadruped can't track it. We rely on yaw
        to redirect the body's forward-walk so that walking straight
        eventually reaches ideal_body.
        """
        # 1. Choose the yaw at which body-forward extension lands tip on cube.
        ee_lat_angle = float(
            np.arctan2(self._ee_offset_body[1], self._ee_offset_body[0])
        )
        yaw_to_cube = np.arctan2(
            cube_world[1] - base_pos[1], cube_world[0] - base_pos[0]
        )
        target_yaw = yaw_to_cube - ee_lat_angle

        # 2. The ideal body position, given target_yaw.
        c, s = np.cos(target_yaw), np.sin(target_yaw)
        eo = self._ee_offset_body
        ideal_xy = np.array(
            [
                cube_world[0] - (c * eo[0] - s * eo[1]),
                cube_world[1] - (s * eo[0] + c * eo[1]),
            ]
        )

        # 3. Body→ideal in body frame (along current yaw, for cmd_vel).
        ddx = float(ideal_xy[0] - base_pos[0])
        ddy = float(ideal_xy[1] - base_pos[1])
        c_, s_ = np.cos(-base_yaw), np.sin(-base_yaw)
        body_forward_err = c_ * ddx - s_ * ddy

        # 4. Yaw: drive toward target_yaw.
        yaw_err = (target_yaw - base_yaw + np.pi) % (2 * np.pi) - np.pi
        self.cmd_vel[2] = float(np.clip(yaw_err * 1.5, -SERVO_MAX_WZ, SERVO_MAX_WZ))

        # 5. Forward: keep walking forward as long as IK can't actually reach
        # the cube. The "ideal_body" position assumes the arm will hit
        # workspace exactly — but at off-axis positions the IK plateaus 5+cm
        # short, so body needs to creep CLOSER than ideal_body to close that
        # gap. Stop only when ik_err is genuinely small (arm physically
        # reaching the bar, not just at the static-reach-pose distance).
        ik_err = float(self._last_ik_err) if not np.isnan(self._last_ik_err) else 1.0
        # Hard floor on body→target distance: dog's nose is +0.285m from base,
        # so anything closer than ~0.40m means base→nose collides with target.
        d_body_to_goal = float(np.linalg.norm(cube_world[:2] - base_pos[:2]))
        if d_body_to_goal < 0.32:
            # Don't approach further — let arm IK do the rest.
            self.cmd_vel[0] = 0.0
        elif ik_err > SERVO_TOL_M:
            # IK can't reach cube yet → press body forward.
            self.cmd_vel[0] = SERVO_MIN_VX
        elif abs(body_forward_err) > 0.04:
            # IK reaches cube; fine-position body to ideal_body.
            sign = 1.0 if body_forward_err > 0 else -1.0
            v = sign * max(SERVO_MIN_VX, SERVO_KP * abs(body_forward_err))
            self.cmd_vel[0] = float(np.clip(v, -SERVO_MAX_VX, SERVO_MAX_VX))
        else:
            self.cmd_vel[0] = 0.0
        self.cmd_vel[1] = 0.0

    def _build_obs(self, model, data):
        gyro = data.sensor("gyro").data.copy()
        imu_xmat = data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_pos = data.qpos[self._qpos_indices] - DEFAULT_OFFSETS
        joint_vel = data.qvel[self._qvel_indices] * JOINT_VEL_SCALE
        return np.concatenate(
            [
                gyro * ANG_VEL_SCALE,
                gravity,
                self.cmd_vel,
                np.clip(joint_pos, -100, 100),
                np.clip(joint_vel, -100, 100),
                self._last_action,
            ]
        ).astype(np.float32)

    # --- main control callback ----
    def get_control(self, model, data):
        self._counter += 1
        if self._counter % self._n_substeps != 0:
            return

        # 1) Vision at ~10Hz with goal LOCKING:
        #   - First N detections are accumulated (while robot is far and
        #     sees the full object) and averaged → locked goal
        #   - Once locked, ignore close-range detections (which see only
        #     part of the object and jitter); only unlock if the new
        #     detection differs by > GOAL_MOVED_THRESHOLD (cube moved).
        if (data.time - self._last_vision_t) >= 0.10:
            det, self.last_rgb = self.vision.detect(data)
            if det is not None:
                new_goal = det["goal_world"]
                is_partial = bool(det.get("partial", False))
                # `_last_detect_t` is the watchdog for "do we have ANY recent
                # signal about the object" — partial detections still count
                # (we know it's there, just can't fix grasp point on it).
                self._last_detect_t = data.time
                if is_partial:
                    # Bbox touches image edge → only seeing part of the object,
                    # grasp point would land on an arbitrary visible chunk
                    # (e.g. one plate). Don't update the goal.
                    pass
                elif not self._goal_locked:
                    self._goal_lock_buf.append(new_goal)
                    self.goal_world = np.mean(self._goal_lock_buf, axis=0).astype(
                        np.float32
                    )
                    if len(self._goal_lock_buf) == 1:
                        tw = det.get("thin_width_m", float("nan"))
                        tp = det.get("thin_pixels", -1)
                        print(
                            f"[vision] first detect: thin ≈ {tw*100:.1f}cm "
                            f"({tp}px), graspable={det.get('graspable')} "
                            f"(jaws open {GRIPPER_OPEN_M*100:.0f}cm)"
                        )
                    if len(self._goal_lock_buf) >= self.GOAL_LOCK_SAMPLES:
                        self._goal_locked = True
                        print(
                            f"[vision] 🔒 goal LOCKED at {self.goal_world.round(3)} "
                            f"(averaged over {self.GOAL_LOCK_SAMPLES} detections)"
                        )
                elif self.phase in ("SEARCH", "APPROACH"):
                    # Locked AND full view: only react if cube has actually
                    # moved. Partial views (handled above) never re-lock,
                    # which prevents close-range "I only see one plate"
                    # detections from corrupting the lock. Restricted to
                    # SEARCH/APPROACH so that mid-grasp drift (e.g. our own
                    # arm bumping the bar during SERVO/CLOSE) can't rewrite
                    # the target out from under the controller.
                    drift = float(np.linalg.norm(new_goal - self.goal_world))
                    if drift > self.GOAL_MOVED_THRESHOLD:
                        print(f"[vision] 🔓 goal moved ({drift:.2f}m) → re-locking")
                        self._goal_lock_buf = [new_goal]
                        self._goal_locked = False
                        self.goal_world = new_goal.astype(np.float32)
            self._last_vision_t = data.time
        vision_age = (
            (data.time - self._last_detect_t)
            if self._last_detect_t >= 0
            else float("inf")
        )

        # 2) Read live state
        base_pos = data.qpos[0:3].copy()
        base_yaw = self._quat_yaw(data.qpos[3:7].copy())
        ee_pos = self._ee_tip_world(data)

        # 3) State machine
        # Visual servo design: arm extends to REACH_POSE, then body cmd_vel
        # is driven by EE→cube error to bring EE onto the cube. Close-range
        # phases (SERVO/CLOSE/LIFT_TEST) tolerate vision loss — cube falling
        # out of FOV at close range is expected and the cached goal still
        # holds in world frame.
        prev_phase = self.phase
        # Vision-loss watchdog: only fires BEFORE the goal is locked (still
        # accumulating samples). Once LOCKED we trust the cached world goal
        # and don't care if the camera momentarily loses the cube — APPROACH
        # walks toward an approach_xy that may take the cube outside FOV
        # (e.g. for off-axis cubes the body yaws sharply and the camera
        # stops pointing at the cube). Re-locking from those partial views
        # would corrupt the goal anyway (already filtered out elsewhere).
        if (
            self.phase == "APPROACH"
            and not self._goal_locked
            and vision_age > VISION_LOST_TIMEOUT
        ):
            print(
                f"[vision lost in APPROACH] backing up to reacquire (vision_age={vision_age:.1f}s)"
            )
            self.phase, self.phase_t = "REACQUIRE", 0.0

        if self.phase == "REACQUIRE":
            # If vision found cube fresh (within last 0.3s), reacquired → APPROACH
            if vision_age < 0.3:
                print(f"[reacquired] cube back in view, resuming APPROACH")
                self.phase, self.phase_t = "APPROACH", 0.0
            elif self.phase_t > REACQUIRE_TIMEOUT_S:
                # Backing up didn't help → cube probably truly gone, full SEARCH
                print(
                    f"[reacquire failed] {REACQUIRE_TIMEOUT_S}s of backing up didn't bring cube back → SEARCH"
                )
                self.goal_world = None
                self.phase, self.phase_t = "SEARCH", 0.0

        if self.goal_world is None:
            self.phase = "SEARCH"
        elif self.phase == "REACQUIRE":
            pass  # handled above
        else:
            d_ee_to_goal = float(np.linalg.norm(self.goal_world - ee_pos))

            if self.phase == "SEARCH":
                self.phase, self.phase_t = "APPROACH", 0.0
            elif self.phase == "APPROACH":
                # Walk toward approach_xy (bar-perpendicular standoff). Then
                # PRE_ALIGN to fix yaw to bar-perpendicular before EXTEND.
                approach_xy, _, _ = self._compute_approach_pose(
                    self.goal_world, base_pos
                )
                self._approach_xy = approach_xy
                d_to_approach = float(np.linalg.norm(approach_xy - base_pos[:2]))
                if d_to_approach < PRE_ALIGN_XY_TOL:
                    self.phase, self.phase_t = "PRE_ALIGN", 0.0
            elif self.phase == "PRE_ALIGN":
                # At approach_xy, rotate body to target_yaw (bar-perpendicular).
                _, target_yaw, _ = self._compute_approach_pose(
                    self.goal_world, base_pos
                )
                self._approach_yaw = target_yaw
                yaw_err = abs((target_yaw - base_yaw + np.pi) % (2 * np.pi) - np.pi)
                if yaw_err < PRE_ALIGN_YAW_TOL:
                    # Now compute reach pose at this LIVE perpendicular pose.
                    (
                        self._reach_pose,
                        self._ee_offset_body,
                        self._reach_target_world,
                    ) = self._compute_reach_pose(
                        self._model,
                        target_world_z=float(self.goal_world[2]),
                        live_data=data,
                    )
                    self._prev_reach_pose = self._reach_pose.copy()
                    self._extend_ee_start = self._ee_tip_world(data).copy()
                    print(
                        f"  reach pose for target z={self.goal_world[2]:.3f}: "
                        f"j={self._reach_pose.round(3)}, ee_body={self._ee_offset_body.round(3)}"
                    )
                    self.phase, self.phase_t = "EXTEND", 0.0
                elif self.phase_t > PRE_ALIGN_TIMEOUT:
                    print(
                        f"[PRE_ALIGN] timed out, yaw_err={yaw_err:.2f}rad → EXTEND anyway"
                    )
                    (
                        self._reach_pose,
                        self._ee_offset_body,
                        self._reach_target_world,
                    ) = self._compute_reach_pose(
                        self._model,
                        target_world_z=float(self.goal_world[2]),
                        live_data=data,
                    )
                    self._prev_reach_pose = self._reach_pose.copy()
                    self._extend_ee_start = self._ee_tip_world(data).copy()
                    self.phase, self.phase_t = "EXTEND", 0.0
            elif self.phase == "EXTEND":
                # All joints ramp simultaneously over EXTEND_RAMP_S.
                if self.phase_t > EXTEND_RAMP_S:
                    self.phase, self.phase_t = "SERVO", 0.0
                    self._servo_best_d = d_ee_to_goal
            elif self.phase == "SERVO":
                # Drive EE to the bar's central axis directly. The jaws at
                # J4_OPEN straddle that point, so on CLOSE the bar ends up
                # clamped between them.
                _, _, perp = self._compute_approach_pose(self.goal_world, base_pos)
                ik_target = self.goal_world + np.array(
                    [
                        perp[0] * GRIPPER_OFFSET_BACK,
                        perp[1] * GRIPPER_OFFSET_BACK,
                        GRIPPER_OFFSET_UP,
                    ],
                    dtype=np.float32,
                )
                ik_q, ik_err = self.ik.solve(
                    data,
                    ik_target,
                    max_iter=120,
                    damp=0.03,
                    warm_only=True,
                    j1_max=J1_GRASP_MAX,
                )
                self._reach_pose = self._slew_reach_pose(ik_q)
                self._last_ik_err = float(ik_err)
                d_ee_to_target = float(np.linalg.norm(ik_target - ee_pos))
                if d_ee_to_target < self._servo_best_d:
                    self._servo_best_d = d_ee_to_target
                if int(self.phase_t * 5) != int((self.phase_t - self._step_dt) * 5):
                    d_body = float(np.linalg.norm(self.goal_world[:2] - base_pos[:2]))
                    print(
                        f"  [SERVO] t={self.phase_t:.1f}s  d_body={d_body:.3f}m  "
                        f"d_ee_target={d_ee_to_target:.3f}m  ik_err={ik_err:.3f}m  "
                        f"cmd_v=({self.cmd_vel[0]:+.2f},{self.cmd_vel[1]:+.2f},{self.cmd_vel[2]:+.2f})"
                    )
                if d_ee_to_target < SERVO_TOL_M:
                    print(
                        f"[SERVO] reached tol: d_ee_target={d_ee_to_target:.3f}m "
                        f"(ik_err={ik_err:.3f}m)"
                    )
                    self.phase, self.phase_t = "CLOSE", 0.0
                elif self.phase_t > SERVO_TIMEOUT_S:
                    print(
                        f"[SERVO] timed out, best d={self._servo_best_d:.3f}m "
                        f"(last IK err={self._last_ik_err:.3f}m) → closing"
                    )
                    self.phase, self.phase_t = "CLOSE", 0.0
            elif self.phase == "CLOSE":
                if self.phase_t > CLOSE_HOLD_S:
                    # Set up the active lift test: snapshot cube z, solve IK
                    # for "current EE position but LIFT_HEIGHT_M higher" so we
                    # pull straight up from where we grabbed (no XY drift that
                    # would yank the cube sideways).
                    self._lift_cube_z_start = float(
                        data.geom_xpos[self._handle_geom_id][2]
                    )
                    self._lift_from_pose = self._reach_pose.copy()
                    lift_target = ee_pos.copy()
                    lift_target[2] += LIFT_HEIGHT_M
                    self._lift_reach_pose, lift_err = self.ik.solve(
                        data,
                        lift_target,
                        max_iter=120,
                        damp=0.03,
                        j1_max=J1_GRASP_MAX,
                    )
                    if lift_err > 0.04:
                        print(
                            f"  [lift] IK residual {lift_err:.3f}m — arm at "
                            f"workspace limit, partial lift only"
                        )
                    self._lift_contact_seen = False
                    self.phase, self.phase_t = "LIFT_TEST", 0.0
            elif self.phase == "LIFT_TEST":
                # While the arm is rising, accumulate whether contact was
                # observed at any point — gripper jaws aren't perfect, but a
                # truly-held cube will register contact for several frames.
                if self._cube_in_gripper_contact(model, data):
                    self._lift_contact_seen = True
                if self.phase_t > (LIFT_RAMP_S + LIFT_HOLD_S):
                    handle_pos = data.geom_xpos[self._handle_geom_id].copy()
                    dz = float(handle_pos[2]) - self._lift_cube_z_start
                    d_cube_ee = float(np.linalg.norm(handle_pos - ee_pos))
                    # Honest grasp: cube actually moved up with the arm.
                    # Distance check is just a sanity gate (cube hasn't been
                    # knocked far away).
                    held = (dz > LIFT_DETECT_M) and (d_cube_ee < GRASP_HOLD_RADIUS)
                    self.grasp_success = held
                    if held:
                        print(
                            f"[LIFT_TEST] ✓ HELD (cube rose {dz*100:+.1f}cm, "
                            f"d_cube_ee={d_cube_ee:.3f}m, contact_seen={self._lift_contact_seen})"
                        )
                        self.phase, self.phase_t = "DONE", 0.0
                    else:
                        self._retry_count += 1
                        if self._retry_count > RETRY_MAX:
                            print(
                                f"[LIFT_TEST] ✗ MISSED #{self._retry_count} "
                                f"— exhausted {RETRY_MAX} retries, giving up"
                            )
                            self.phase, self.phase_t = "DONE", 0.0
                        else:
                            print(
                                f"[LIFT_TEST] ✗ MISSED #{self._retry_count} "
                                f"(cube Δz={dz*100:+.1f}cm < {LIFT_DETECT_M*100:.1f}cm "
                                f"or d_cube_ee={d_cube_ee:.3f}m too far) → backing up to retry"
                            )
                            # The failed grasp likely bumped the cube. Unlock the goal
                            # so vision can detect its new position.
                            self._goal_locked = False
                            self._goal_lock_buf = []
                            self.phase, self.phase_t = "RETRY_BACKUP", 0.0
            elif self.phase == "RETRY_BACKUP":
                if self.phase_t > RETRY_BACKUP_S:
                    # Reset reach_pose to a fresh PRE_ALIGN-style IK solution
                    # — otherwise it's still the SERVO-end pose of the failed
                    # attempt, and the next EXTEND ramp would interpolate
                    # HOLD → that weird pose.
                    (
                        self._reach_pose,
                        self._ee_offset_body,
                        self._reach_target_world,
                    ) = self._compute_reach_pose(
                        self._model,
                        target_world_z=float(self.goal_world[2]),
                        live_data=data,
                    )
                    self._prev_reach_pose = self._reach_pose.copy()
                    # The cube likely moved during the failed grasp, so go back to APPROACH
                    # to walk to the new approach_xy (PRE_ALIGN assumes we are already at
                    # the correct standoff distance, which may no longer be true).
                    self.phase, self.phase_t = "APPROACH", 0.0
        if self.phase != prev_phase:
            extra = (
                f" goal={self.goal_world.round(3)}"
                if self.goal_world is not None
                else ""
            )
            print(f"[{prev_phase} → {self.phase}]{extra}")

        # 4) cmd_vel per phase
        if self.phase == "SEARCH":
            # No prior detection — slow yaw rotation to scan for object.
            self.cmd_vel[0] = 0.0
            self.cmd_vel[1] = 0.0
            self.cmd_vel[2] = SEARCH_YAW_RATE
        elif self.phase == "REACQUIRE":
            # Cube was seen before but currently out of FOV — back up so it
            # falls back into view (most common cause: we got too close).
            self.cmd_vel[0] = REACQUIRE_BACK_VX
            self.cmd_vel[1] = 0.0
            self.cmd_vel[2] = 0.0
        elif self.phase == "RETRY_BACKUP":
            # Visibly "try again": back up + alternate lateral side-step so
            # next attempt comes from a slightly different angle.
            self.cmd_vel[0] = -0.20
            # alternate left / right based on retry count so we wobble
            sign = +1 if (self._retry_count % 2 == 0) else -1
            self.cmd_vel[1] = float(sign * RETRY_SIDE_STEP)
            self.cmd_vel[2] = 0.0
        elif self.phase == "APPROACH" and self.goal_world is not None:
            # Walk toward approach_xy with yaw + forward + small lateral.
            dx = float(self._approach_xy[0] - base_pos[0])
            dy = float(self._approach_xy[1] - base_pos[1])
            c, s = np.cos(-base_yaw), np.sin(-base_yaw)
            body_x = c * dx - s * dy
            body_y = s * dx + c * dy
            norm = max(np.hypot(body_x, body_y), 1e-6)
            speed = float(np.clip(0.6 * norm, 0.20, 0.5))
            self.cmd_vel[0] = float(speed * body_x / norm)
            self.cmd_vel[1] = float(speed * body_y / norm) * 0.3
            yaw_to_target = np.arctan2(dy, dx)
            yaw_err = (yaw_to_target - base_yaw + np.pi) % (2 * np.pi) - np.pi
            self.cmd_vel[2] = float(np.clip(yaw_err * 1.5, -1.0, 1.0))
        elif self.phase == "PRE_ALIGN" and self.goal_world is not None:
            # At approach_xy — rotate yaw to perpendicular target. No forward.
            yaw_err = (self._approach_yaw - base_yaw + np.pi) % (2 * np.pi) - np.pi
            self.cmd_vel[0] = 0.0
            self.cmd_vel[1] = 0.0
            self.cmd_vel[2] = float(np.clip(yaw_err * 1.5, -SERVO_MAX_WZ, SERVO_MAX_WZ))
        elif self.phase == "SERVO" and self.goal_world is not None:
            # cmd_vel is driven by IK residual inside _set_servo_command:
            # walks forward only while arm can't reach, freezes once it can.
            self._set_servo_command(base_pos, base_yaw, self.goal_world)
        elif self.phase in ("EXTEND", "CLOSE", "LIFT_TEST", "DONE"):
            self.cmd_vel[:] = 0.0
        else:
            self.cmd_vel[:] = 0.0

        # 5) joint4 (gripper)
        # CLOSE / LIFT_TEST: gripper closed (must stay closed during lift to
        # actually carry the cube). DONE: stay closed only if grasp succeeded.
        # everything else (incl. RETRY_BACKUP): gripper open
        if self.phase in ("CLOSE", "LIFT_TEST"):
            j4_target = J4_CLOSE
        elif self.phase == "DONE" and self.grasp_success:
            j4_target = J4_CLOSE
        else:
            j4_target = J4_OPEN

        # 6) Walk policy → 12 leg targets (always)
        obs = self._build_obs(model, data)
        action = self.policy.run(self._output_names, {"obs": obs.reshape(1, -1)})[0][0]
        self._last_action = action.copy()
        walk_targets = (action * ACTION_SCALE + DEFAULT_OFFSETS).astype(np.float32)

        # 7) DIY targets
        # APPROACH: HOLD pose; EXTEND: ramp HOLD→REACH_POSE; SERVO/CLOSE:
        # REACH_POSE; LIFT_TEST: ramp REACH_POSE → LIFT_REACH_POSE so the arm
        # rises while gripper stays closed (active grasp test).
        # joint2/joint3 dispatch (j1 handled separately below)
        fl_diy_targets = np.empty(4, dtype=np.float32)
        if self.phase in (
            "SEARCH",
            "APPROACH",
            "PRE_ALIGN",
            "REACQUIRE",
            "RETRY_BACKUP",
        ):
            # Hold-like phases: arm folded back to leg-tuck.
            fl_diy_targets[:3] = FL_DIY_HOLD_123
        elif self.phase == "EXTEND":
            # Task-space interpolation: EE follows a straight line in world
            # coords from its HOLD-pose position to the reach target, with a
            # smoothstep ease. IK each tick yields the joint command.
            # j1_max ramps from J1_RETRACTED → J1_GRASP_MAX in sync, so the
            # slider extends gradually instead of snapping to the cap on the
            # first tick (which used to push j2/j3 straight to their final
            # values, producing the "lift first" artifact).
            a = float(np.clip(self.phase_t / EXTEND_RAMP_S, 0.0, 1.0))
            s = a * a * (3.0 - 2.0 * a)
            ee_target = (
                1 - s
            ) * self._extend_ee_start + s * self._reach_target_world
            j1_max_ramp = (1 - s) * J1_RETRACTED + s * J1_GRASP_MAX
            ik_q, _ = self.ik.solve(
                data,
                ee_target,
                max_iter=80,
                damp=0.03,
                warm_only=True,
                j1_max=j1_max_ramp,
            )
            fl_diy_targets[:3] = ik_q
        elif self.phase == "LIFT_TEST":
            a = float(np.clip(self.phase_t / LIFT_RAMP_S, 0.0, 1.0))
            fl_diy_targets[:3] = (
                1 - a
            ) * self._lift_from_pose + a * self._lift_reach_pose
        else:  # SERVO, CLOSE, DONE → REACH_POSE
            fl_diy_targets[:3] = self._reach_pose

        # joint1 prismatic slider:
        # - EXTEND: task-space IK above already produced fl_diy_targets[0].
        # - SERVO/CLOSE/LIFT_TEST: IK direct.
        # - DONE/non-grasp: slew back to retracted (10cm) at 3cm/s.
        if self.phase == "EXTEND":
            self._j1_cmd = float(fl_diy_targets[0])
        elif self.phase in ("SERVO", "CLOSE"):
            fl_diy_targets[0] = float(self._reach_pose[0])
            self._j1_cmd = float(self._reach_pose[0])
        elif self.phase == "LIFT_TEST":
            fl_diy_targets[0] = float(self._lift_reach_pose[0])
            self._j1_cmd = float(self._lift_reach_pose[0])
        else:
            # SEARCH / APPROACH / PRE_ALIGN / REACQUIRE / RETRY_BACKUP / DONE
            # → slew back to retracted (10cm).
            max_step = J1_SLEW_RATE * self._step_dt
            if self._j1_cmd < J1_RETRACTED:
                self._j1_cmd = min(J1_RETRACTED, self._j1_cmd + max_step)
            else:
                self._j1_cmd = max(J1_RETRACTED, self._j1_cmd - max_step)
            fl_diy_targets[0] = self._j1_cmd
        fl_diy_targets[3] = j4_target

        # Slew-limit j2/j3 so phase transitions (e.g. LIFT_TEST→RETRY_BACKUP,
        # CLOSE→DONE on failed grasp) don't snap the arm from extended pose
        # to HOLD in one tick. j1 has its own dedicated slew above.
        max_j23_step = JOINT_SLEW_RATE * self._step_dt
        for k in (1, 2):
            delta = float(fl_diy_targets[k] - self._last_arm_target_j23[k - 1])
            fl_diy_targets[k] = self._last_arm_target_j23[k - 1] + np.clip(
                delta, -max_j23_step, max_j23_step
            )
        self._last_arm_target_j23[:] = fl_diy_targets[1:3]

        # 7) Write actuators
        for i in range(12):
            data.ctrl[self._actuator_indices[i]] = walk_targets[i]
        for i in range(4):
            data.ctrl[self._fl_diy_act_ids[i]] = fl_diy_targets[i]
        for i in range(4):
            data.ctrl[self._fr_diy_act_ids[i]] = self._fr_diy_hold[i]

        self.phase_t += self._step_dt


# =========================================================================
# Entry point
# =========================================================================
def _rerun_setup(spawn: bool = True):
    """Initialize rerun viewer. Spawns the GUI if `rerun` binary is on PATH;
    otherwise falls back to streaming to the default tcp port (open viewer
    manually with `rerun --connect`).
    """
    rr = _get_rerun()
    # Silence rerun's noisy gRPC-shutdown ERROR lines that always fire when
    # we Ctrl+C — they're harmless but spam the terminal.
    os.environ.setdefault("RUST_LOG", "rerun=warn,re_grpc_client=off,re_sdk=off")
    rr.init("walk_and_grasp")
    if spawn:
        try:
            rr.spawn()
        except Exception as e:
            print(f"[rerun] couldn't auto-spawn viewer ({e.__class__.__name__})")
            print("[rerun] streaming to localhost:9876 — open viewer manually:")
            print("        conda activate sim2sim && rerun --connect")
            rr.serve_web(open_browser=False) if hasattr(rr, "serve_web") else None
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def _rerun_log(rr_module, ctrl: GraspController, data: mujoco.MjData):
    """Log one frame of state to rerun. Cheap — call ~20Hz."""
    rr = rr_module
    t = float(data.time)
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("sim_time", t)
    else:
        rr.set_time("sim_time", duration=t)

    # State machine phase as a text log
    rr.log("phase", rr.TextLog(ctrl.phase, level=rr.TextLogLevel.INFO))

    # 3D view: base, EE, goal, handle ground-truth
    base_pos = data.qpos[0:3].copy()
    ee_pos = ctrl._ee_tip_world(data)
    handle_pos = data.geom_xpos[ctrl._handle_geom_id].copy()
    rr.log(
        "world/base", rr.Points3D([base_pos], colors=[[100, 150, 255]], radii=[0.04])
    )
    rr.log("world/ee", rr.Points3D([ee_pos], colors=[[255, 200, 0]], radii=[0.025]))
    rr.log(
        "world/cube_truth",
        rr.Points3D([handle_pos], colors=[[255, 50, 50]], radii=[0.025]),
    )
    if ctrl.goal_world is not None:
        rr.log(
            "world/vision_goal",
            rr.Points3D([ctrl.goal_world], colors=[[50, 255, 50]], radii=[0.025]),
        )
        rr.log(
            "world/ee_to_goal_line",
            rr.LineStrips3D([[ee_pos, ctrl.goal_world]], colors=[[255, 255, 100]]),
        )

    # Camera image with detection overlay
    if ctrl.last_rgb is not None:
        rr.log("camera/rgb", rr.Image(ctrl.last_rgb))

    # Scalars over time
    if ctrl.goal_world is not None:
        d_ee = float(np.linalg.norm(ctrl.goal_world - ee_pos))
        d_cube_ee = float(np.linalg.norm(handle_pos - ee_pos))
        rr.log("scalars/d_ee_to_goal", rr.Scalars(d_ee))
        rr.log("scalars/d_cube_to_ee", rr.Scalars(d_cube_ee))
        rr.log("scalars/cmd_vel_x", rr.Scalars(float(ctrl.cmd_vel[0])))
        rr.log("scalars/cmd_vel_yaw", rr.Scalars(float(ctrl.cmd_vel[2])))


def _draw_overlays(scene, ctrl: GraspController, data: mujoco.MjData):
    """Goal marker (red sphere + line) drawn each frame in passive viewer."""
    scene.ngeom = 0
    geom_idx = 0

    if ctrl.goal_world is not None and scene.maxgeom > geom_idx:
        g = np.asarray(ctrl.goal_world, dtype=np.float64)
        mujoco.mjv_initGeom(
            scene.geoms[geom_idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.015, 0, 0], dtype=np.float64),
            pos=g,
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array([0.1, 0.9, 0.1, 0.8], dtype=np.float32),
        )
        geom_idx += 1

    if scene.maxgeom > geom_idx:
        ee = ctrl._ee_tip_world(data)
        mujoco.mjv_initGeom(
            scene.geoms[geom_idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.015, 0, 0], dtype=np.float64),
            pos=ee,
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array([0.9, 0.8, 0.1, 0.8], dtype=np.float32),
        )
        geom_idx += 1

    scene.ngeom = geom_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=_ONNX_PATH)
    p.add_argument(
        "--cube",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override target_cube spawn position (default: keyframe value)",
    )
    p.add_argument(
        "--headless", action="store_true", help="Run without viewer (for smoke testing)"
    )
    p.add_argument(
        "--duration", type=float, default=20.0, help="Headless run duration (s)"
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Stream telemetry to rerun viewer (camera, 3D pose, scalars)",
    )
    p.add_argument(
        "--record",
        metavar="FILE",
        help="Render the scene offscreen and write an MP4 to FILE",
    )
    p.add_argument(
        "--record-fps",
        type=int,
        default=30,
        help="Frames per second for --record (default 30)",
    )
    p.add_argument(
        "--record-size",
        type=int,
        nargs=2,
        default=[960, 720],
        metavar=("W", "H"),
        help="Recorded video resolution",
    )
    args = p.parse_args()

    print("=" * 64)
    print("  Walk + Vision + Grasp")
    print("=" * 64)

    model = mujoco.MjModel.from_xml_path(_MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    if args.cube is not None:
        cube_jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_free"
        )
        cube_adr = model.jnt_qposadr[cube_jid]
        data.qpos[cube_adr : cube_adr + 3] = args.cube
        data.qpos[cube_adr + 3 : cube_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        print(f"  cube re-spawned at {args.cube}")

    ctrl = GraspController(model, args.policy)
    for i in range(12):
        data.ctrl[ctrl._actuator_indices[i]] = DEFAULT_OFFSETS[i]
    mujoco.set_mjcb_control(ctrl.get_control)

    rr = None
    if args.rerun:
        _rerun_setup(spawn=True)
        rr = _get_rerun()
        print("rerun viewer enabled")
    last_rerun_t = -1.0

    # --record: stream offscreen 3rd-person frames to an MP4 via imageio.
    rec_renderer = None
    rec_writer = None
    rec_cam = None
    rec_dt = None
    last_rec_t = -1.0
    rec_path = args.record
    if rec_path:
        import imageio

        W, H = int(args.record_size[0]), int(args.record_size[1])
        rec_renderer = mujoco.Renderer(model, height=H, width=W)
        rec_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, rec_cam)
        rec_cam.distance = 2.0
        rec_cam.azimuth = 135.0
        rec_cam.elevation = -25.0
        rec_writer = imageio.get_writer(
            rec_path, fps=int(args.record_fps), codec="libx264", quality=8
        )
        rec_dt = 1.0 / float(args.record_fps)
        print(f"[record] writing {W}x{H} @ {args.record_fps}fps → {rec_path}")

    def _capture_frame():
        nonlocal last_rec_t
        if rec_writer is None:
            return
        if (data.time - last_rec_t) < rec_dt:
            return
        last_rec_t = data.time
        # Track the dog so the action stays in frame
        rec_cam.lookat[:] = data.qpos[0:3]
        rec_renderer.update_scene(data, camera=rec_cam)
        frame = rec_renderer.render()
        rec_writer.append_data(frame)

    try:
        if args.headless or rec_path:
            n = int(args.duration / model.opt.timestep)
            for _ in range(n):
                mujoco.mj_step(model, data)
                if rr is not None and (data.time - last_rerun_t) > 0.05:
                    _rerun_log(rr, ctrl, data)
                    last_rerun_t = data.time
                _capture_frame()
            print(f"\nfinal phase = {ctrl.phase}")
            print(f"grasp_success = {ctrl.grasp_success}")
            print(
                f"final base z = {data.qpos[2]:.3f}m  ({'upright' if data.qpos[2] > 0.18 else 'COLLAPSED'})"
            )
            if ctrl.goal_world is not None:
                ee = ctrl._ee_tip_world(data)
                handle = data.geom_xpos[ctrl._handle_geom_id].copy()
                d = float(np.linalg.norm(ee - ctrl.goal_world))
                d_handle = float(np.linalg.norm(ee - handle))
                print(f"final EE→goal = {d:.3f}m,  EE→handle = {d_handle:.3f}m")
        else:
            with viewer.launch_passive(model, data) as v:
                v.cam.distance = 2.5
                v.cam.azimuth = 135
                v.cam.elevation = -25
                wall_t0 = time.time()
                sim_t0 = data.time
                while v.is_running():
                    mujoco.mj_step(model, data)
                    _draw_overlays(v.user_scn, ctrl, data)
                    v.sync()
                    if rr is not None and (data.time - last_rerun_t) > 0.05:
                        _rerun_log(rr, ctrl, data)
                        last_rerun_t = data.time
                    sleep = (data.time - sim_t0) - (time.time() - wall_t0)
                    if sleep > 0:
                        time.sleep(sleep)
    finally:
        mujoco.set_mjcb_control(None)
        if rec_writer is not None:
            rec_writer.close()
            print(f"[record] saved {rec_path}")


if __name__ == "__main__":
    main()
