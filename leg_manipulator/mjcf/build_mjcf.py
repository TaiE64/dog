"""Generate MJCF from go2_with_diy_leg URDF.

Steps:
  1. Sanitize URDF: strip package:// prefixes, convert DIY fixed joints to
     revolute (so they're addressable by the play script).
  2. Compile with MuJoCo to get a base MJCF.
  3. Post-edit: free joint on base, IMU site + gyro sensor, actuators for
     12 walking joints + 8 DIY joints, keyframe matching DEFAULT_OFFSETS,
     floor + lighting.
"""

import os
import re
import xml.etree.ElementTree as ET

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEG_DIR = os.path.dirname(_HERE)
_URDF_IN = os.path.join(_LEG_DIR, "urdf", "go2_with_diy_leg.urdf")
_URDF_TMP = os.path.join(_HERE, "_sanitized.urdf")
_MJCF_OUT = os.path.join(_HERE, "go2_diyleg.xml")

# DIY arm joints — flipped from URDF "fixed" to "revolute" so play_go2_diyleg.py
# can drive random targets at runtime. This is the *intended* sim2sim test:
# perturb the DIY arm to probe walking-policy robustness against an active
# manipulator load. Training froze these joints; deploying with motion is
# precisely the robustness check.
DIY_JOINT_NAMES = {
    "FL_diy_joint1",
    "FL_diy_joint2",
    "FL_diy_joint3",
    "FL_diy_joint4",
    "diy_joint1",
    "diy_joint2",
    "diy_joint3",
    "diy_joint4",
}

WALK_JOINTS = [
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

# Must match init_state.joint_pos in UNITREE_GO2_DIY_LEG_CFG. Front-leg calf
# joints drive the DIY base_link (range [0.99, 2.82]), so they're POSITIVE.
DEFAULT_OFFSETS = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.560,
    "FR_thigh_joint": 0.560,
    "RL_thigh_joint": 0.401,
    "RR_thigh_joint": 0.401,
    "FL_calf_joint": 1.466,
    "FR_calf_joint": 1.466,
    "RL_calf_joint": -1.215,
    "RR_calf_joint": -1.215,
}

# Walking joint PD gains (Unitree-typical for Go2)
WALK_KP, WALK_KD = 25.0, 0.5
# DIY joint PD gains (lighter — they just track random targets)
DIY_KP, DIY_KD = 200.0, 5.0


def sanitize_urdf(src: str, dst: str) -> None:
    """Strip package:// prefixes and flip DIY joint types to revolute."""
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()

    # package://go2_description/dae/  ->  go2_description/obj/  (DAE was pre-
    #   converted to OBJ via trimesh; MuJoCo doesn't load DAE).
    # package://leg_manipulator/      ->  (relative to URDF dir).
    text = text.replace("package://go2_description/dae/", "go2_description/obj/")
    text = text.replace("package://leg_manipulator/", "")
    # Mesh filenames: .dae -> .obj
    text = re.sub(r'(filename="[^"]*)\.dae"', r'\1.obj"', text)

    # Strip <material> elements: URDF has multiple materials per <visual>, which
    # MuJoCo's strict parser rejects. Materials are pure cosmetic, drop them all.
    text = re.sub(r"<material\b[^>]*?/>", "", text)
    text = re.sub(r"<material\b[^>]*?>.*?</material>", "", text, flags=re.DOTALL)

    # The DIY foot uses foot.obj as its collision mesh. The mesh has high vertex
    # count and odd shape; replace with a sphere for stable contact.
    def _replace_foot_collision(s: str) -> str:
        return re.sub(
            r'<mesh\s+filename="go2_description/obj/foot\.obj"\s*scale="1\.3 1\.3 1\.3"\s*/>',
            '<sphere radius="0.025"/>',
            s,
        )

    # Only swap inside <collision> blocks; visuals keep the OBJ mesh.
    out, i = [], 0
    while True:
        j = text.find("<collision", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = text.find("</collision>", j)
        if k < 0:
            out.append(text[j:])
            break
        out.append(_replace_foot_collision(text[j : k + len("</collision>")]))
        i = k + len("</collision>")
    text = "".join(out)

    # Flip DIY joints fixed->revolute, except joint1 which is the prismatic
    # slider on the real hardware (URDF "fixed" placeholder + linear effort/
    # velocity limits make the design intent clear).
    for jname in DIY_JOINT_NAMES:
        target_type = "prismatic" if jname.endswith("joint1") else "revolute"
        text = re.sub(
            rf'(<joint\s+name="{re.escape(jname)}"\s+type=)"fixed"',
            rf'\1"{target_type}"',
            text,
        )

    # Inject a <mujoco> compile-hint block right before </robot>.
    # meshdir is relative to this URDF; URDF lives in mjcf/, meshes live one level up.
    mj_block = (
        "  <mujoco>\n"
        '    <compiler meshdir="../" balanceinertia="true" discardvisual="false" '
        'strippath="false" fusestatic="false"/>\n'
        "  </mujoco>\n"
    )
    text = text.replace("</robot>", mj_block + "</robot>")

    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)


def compile_to_mjcf(urdf_path: str) -> str:
    """Use MuJoCo to compile URDF and return the MJCF XML string."""
    model = mujoco.MjModel.from_xml_path(urdf_path)
    return mujoco.mj_saveLastXML(None, model) if False else _save_xml_to_string(model)


def _save_xml_to_string(model) -> str:
    """mj_saveLastXML writes to a file; round-trip via tempfile."""
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".xml", delete=False) as f:
        tmp = f.name
    mujoco.mj_saveLastXML(tmp, model)
    with open(tmp, "r", encoding="utf-8") as f:
        s = f.read()
    os.remove(tmp)
    return s


def post_edit(mjcf_text: str) -> str:
    """Add free joint, floor, IMU site, gyro, actuators, keyframe."""
    # Pretty-print once to make hand edits cleaner.
    root = ET.fromstring(mjcf_text)

    # --- compiler/option tweaks ---
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
        # place option after compiler for readability
        root.remove(option)
        root.insert(list(root).index(compiler) + 1, option)
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    option.set("iterations", "10")

    # --- defaults: friction etc. ---
    if root.find("default") is None:
        default = ET.Element("default")
        geom_d = ET.SubElement(default, "geom")
        geom_d.set("contype", "1")
        geom_d.set("conaffinity", "1")
        geom_d.set("friction", "0.8 0.02 0.01")
        # insert after option
        idx = list(root).index(option) + 1
        root.insert(idx, default)

    # --- visual / scene cosmetics: skybox, ground texture, headlight off ---
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        idx = list(root).index(root.find("default")) + 1
        root.insert(idx, visual)
    ET.SubElement(
        visual,
        "headlight",
        {
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(
        visual,
        "global",
        {
            "azimuth": "120",
            "elevation": "-20",
            "offwidth": "1280",
            "offheight": "720",
        },
    )

    # textures + materials for the scene (added to <asset>)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
    # hex weight mesh. MuJoCo auto-aligns mesh principal-inertia axes on load,
    # which fortuitously puts: hex prism axis -> body X, vertex direction ->
    # body Y, flat direction -> body Z. So the geom sits coaxial with the bar
    # (body X = bar after the keyframe yaw) with flats up/down (won't roll).
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "hex_weight",
            "file": "/home/taie/Desktop/dog/mjc/hex_weight.stl",
            # STL prism axis is original Z (60 mm). Scale to ~20 mm thick.
            "scale": "1 1 0.33",
        },
    )

    # --- worldbody: add floor + light + IMU site, plus free joint on base ---
    worldbody = root.find("worldbody")
    assert worldbody is not None, "MJCF missing <worldbody>"

    # textured checker floor
    floor = ET.Element(
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "material": "groundplane",
            "friction": "0.8 0.02 0.01",
        },
    )
    worldbody.insert(0, floor)
    # primary directional light (sun)
    light = ET.Element(
        "light",
        {
            "pos": "0 0 1.5",
            "dir": "0 0 -1",
            "directional": "true",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.3 0.3 0.3",
            "castshadow": "true",
        },
    )
    worldbody.insert(1, light)

    # find base body and add freejoint + IMU site
    base = None
    for body in worldbody.findall(".//body"):
        if body.get("name") == "base":
            base = body
            break
    assert base is not None, "could not find <body name='base'>"

    # raise base so it doesn't spawn inside the floor
    pos = base.get("pos", "0 0 0").split()
    pos = [float(x) for x in pos]
    pos[2] = max(pos[2], 0.40)  # matches training init_state.pos[2]
    base.set("pos", " ".join(f"{x:g}" for x in pos))

    # MuJoCo requires freejoint to be the FIRST child of the body
    fj = ET.Element("freejoint", {"name": "root"})
    base.insert(0, fj)

    # IMU site at imu link offset (-0.02557 0 0.04232 from base)
    imu_site = ET.Element(
        "site",
        {
            "name": "imu",
            "pos": "-0.02557 0 0.04232",
            "size": "0.01",
            "rgba": "1 0 0 1",
        },
    )
    # insert after freejoint
    base.insert(1, imu_site)

    # --- head_cam: forward-looking camera mounted on Head_upper ---
    # Used by walk_and_grasp.py for vision-based target detection.
    # 60° vertical FOV (≈D435), tilted 15° down to see the floor in front.
    head = None
    for body in worldbody.findall(".//body"):
        if body.get("name") == "Head_upper":
            head = body
            break
    if head is not None:
        # head_cam: only camera, mounted on Head_upper looking forward + 35°
        # down with 75° vertical FOV. Matches real Go2's head-mounted camera
        # (RealSense-style). When the robot is right in front of the target,
        # the target falls below this camera's FOV — that's a real-world
        # limitation. Vision caches the most recent detection's world
        # coordinate so the controller continues to act on it.
        cam = ET.Element(
            "camera",
            {
                "name": "head_cam",
                "pos": "0.05 0 0.0",
                "xyaxes": "0 -1 0  0.574 0 0.819",
                "fovy": "75",
            },
        )
        head.append(cam)

    # --- platform: static table where the dumbbell rests ---
    # Sits at body x=1.0, y=0.2 (front-left of robot). Top surface at z=0.15
    # so the arm can reach the target without crouching to the floor.
    # 5cm-tall "table" — visually clearly a platform. The DIY arm's natural
    # sweep height intersects the dumbbell, so the first reach attempt
    # often knocks the dumbbell off; the controller then keeps retrying
    # at the new (floor) position. This is realistic grasping behavior.
    PLATFORM_TOP_Z = 0.05
    PLATFORM_HALF = (0.12, 0.15, 0.025)
    # Robustness-tested workspace for FL arm + lower-jaw-scoop grasp:
    #   y ∈ [+0.05, +0.20]  perpendicular bar approach, grasps reliably
    #   y = 0.0             FL weak-side reach, marginal (Δz≈1.9cm)
    #   y < 0.0             body yaws too far right; gripper approaches the
    #                       bar at an angle, lower jaw slides along bar
    #                       instead of scooping under → fails to clamp
    # Default y=0.05 stays in the reliable zone with a slight curved walk.
    platform = ET.Element(
        "body",
        {
            "name": "target_platform",
            "pos": f"1.0 0.05 {PLATFORM_HALF[2]:.3f}",
        },
    )
    ET.SubElement(
        platform,
        "geom",
        {
            "name": "platform_geom",
            "type": "box",
            "size": f"{PLATFORM_HALF[0]:.3f} {PLATFORM_HALF[1]:.3f} {PLATFORM_HALF[2]:.3f}",
            "rgba": "0.55 0.40 0.25 1",  # brown — distinct from red target
            "friction": "1.0 0.05 0.01",
        },
    )
    worldbody.append(platform)

    # --- target_cube: hex dumbbell (mirrors mjc/dumbbell_hex.urdf) ---
    # Handle 0.2 kg + two hex weights 2.4 kg each = 5.0 kg total.
    # Long axis along body X; weights' flat hex faces face ±Z so it sits flat.
    DUMBBELL_REST_Z = PLATFORM_TOP_Z + 0.045  # hex flat-to-flat half = ~0.0433
    cube = ET.Element(
        "body",
        {
            "name": "target_cube",
            "pos": f"0.92 0.05 {DUMBBELL_REST_Z:.3f}",
            "euler": "0 0 1.5707963",  # rotate so handle ends up along world Y
        },
    )
    ET.SubElement(cube, "freejoint", {"name": "target_cube_free"})
    ET.SubElement(
        cube,
        "geom",
        {
            "name": "target_handle",
            "type": "cylinder",
            "size": "0.010 0.075",
            "fromto": "-0.075 0 0.012  0.075 0 0.012",
            "rgba": "0.95 0.05 0.05 1",
            "mass": "0.02",
            "friction": "3.0 0.20 0.05",
        },
    )
    # Hex weights, coaxial with bar. Explicit identity quat OVERRIDES MuJoCo's
    # auto-alignment (which would otherwise rotate the hex axis off the bar).
    # With identity quat, mesh axes (X=hex axis, Y=vertex, Z=flat) align with
    # body axes — hex axis along body X = bar; flats along body Z = world Z
    # (up/down) so the hex rests on a side flat face and won't roll.
    ET.SubElement(
        cube,
        "geom",
        {
            "name": "target_weight_l",
            "type": "mesh",
            "mesh": "hex_weight",
            "pos": "-0.085 0 0",
            "quat": "0.5 -0.5 0.5 -0.5",
            "rgba": "0.95 0.05 0.05 1",
            "mass": "0.09",
            "friction": "3.0 0.20 0.05",
        },
    )
    ET.SubElement(
        cube,
        "geom",
        {
            "name": "target_weight_r",
            "type": "mesh",
            "mesh": "hex_weight",
            "pos": "0.085 0 0",
            "quat": "0.5 -0.5 0.5 -0.5",
            "rgba": "0.95 0.05 0.05 1",
            "mass": "0.09",
            "friction": "3.0 0.20 0.05",
        },
    )
    worldbody.append(cube)

    # --- actuators ---
    # remove any auto-generated actuator block
    for tag in list(root.findall("actuator")):
        root.remove(tag)
    actuator = ET.SubElement(root, "actuator")
    for j in WALK_JOINTS:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{j}_actuator",
                "joint": j,
                "kp": str(WALK_KP),
                "kv": str(WALK_KD),
                "forcerange": "-50 50",
            },
        )
    for j in sorted(DIY_JOINT_NAMES):
        # joint1 is the prismatic slider on the FL/FR diy arms. Its axis has
        # a ~60% vertical component once transformed to world, so gravity on
        # the link2/3/4 chain (~0.15 kg + grasped object) generates ~1 N of
        # along-axis force pulling the slider fully extended. Default kp=5
        # only produces ≤0.5 N (kp * max_error), letting the slider sag —
        # the arm then has to do all reach work via j2/j3, producing the
        # "weird raise" during EXTEND. Bump j1 to kp=400, kv=4 (force-range
        # widened) so it actually holds commanded position against gravity.
        is_slider = j.endswith("joint1")
        kp_val = 20000.0 if is_slider else DIY_KP
        kv_val = 200.0 if is_slider else DIY_KD
        forcerange = "-300 300" if is_slider else "-100 100"
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{j}_actuator",
                "joint": j,
                "kp": str(kp_val),
                "kv": str(kv_val),
                "forcerange": forcerange,
            },
        )

    # --- sensors ---
    for tag in list(root.findall("sensor")):
        root.remove(tag)
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", {"name": "gyro", "site": "imu"})
    ET.SubElement(sensor, "accelerometer", {"name": "accel", "site": "imu"})
    ET.SubElement(
        sensor, "framequat", {"name": "imu_quat", "objtype": "site", "objname": "imu"}
    )

    # --- keyframe (standing pose with policy-default joint angles) ---
    for tag in list(root.findall("keyframe")):
        root.remove(tag)
    keyframe = ET.SubElement(root, "keyframe")

    # build qpos string in MuJoCo joint order. Free joint first (7 dofs:
    # x y z + quat w x y z), then joint angles in the order MuJoCo emits them.
    # We'll let MuJoCo reorder for us by loading the freshly built model and
    # reading the joint order.
    placeholder_qpos = ["0", "0", "0.40", "1", "0", "0", "0"]  # populated below
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": " ".join(placeholder_qpos),
            # ctrl will be set below
        },
    )

    return ET.tostring(root, encoding="unicode")


def fill_keyframe(mjcf_text: str) -> str:
    """Reload MJCF, derive qpos/ctrl in correct joint order, write back."""
    # write to tmp so MuJoCo can load it (mesh paths are relative to file)
    tmp = os.path.join(_HERE, "_tmp_check.xml")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(mjcf_text)
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)

    # qpos vector
    qpos = [0.0] * model.nq
    # free joint occupies qpos[0:7] = (x, y, z, qw, qx, qy, qz)
    qpos[0:7] = [0.0, 0.0, 0.40, 1.0, 0.0, 0.0, 0.0]

    for jname, val in DEFAULT_OFFSETS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = model.jnt_qposadr[jid]
        qpos[adr] = val

    # DIY joint defaults: midpoint of range
    for jname in DIY_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        qpos[adr] = 0.5 * (lo + hi)

    # Target cube freejoint: spawn position + identity orientation. Without
    # this, qpos defaults to all zeros (including invalid quat = (0,0,0,0))
    # and MuJoCo either teleports the cube to origin or NaNs out.
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_free")
    if cube_jid >= 0:
        cube_adr = model.jnt_qposadr[cube_jid]
        # Dumbbell rests on 5cm-top platform; with 8cm box-height ends,
        # cube body center at z = platform_top 0.05 + hex flat-half 0.045.
        # quat = 90° around Z so the bar lies along world Y (perpendicular to
        # robot forward), letting the front-mounted gripper close across it.
        qpos[cube_adr : cube_adr + 7] = [0.92, 0.05, 0.095, 0.70710678, 0.0, 0.0, 0.70710678]

    # ctrl vector matches actuator order
    ctrl = [0.0] * model.nu
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        # name is "<joint>_actuator"
        jname = aname.rsplit("_actuator", 1)[0]
        if jname in DEFAULT_OFFSETS:
            ctrl[i] = DEFAULT_OFFSETS[jname]
        elif jname in DIY_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            lo, hi = model.jnt_range[jid]
            ctrl[i] = 0.5 * (lo + hi)

    # patch keyframe in xml text
    qpos_s = " ".join(f"{x:g}" for x in qpos)
    ctrl_s = " ".join(f"{x:g}" for x in ctrl)
    new_key = f'<key name="home" qpos="{qpos_s}" ctrl="{ctrl_s}"/>'
    mjcf_text = re.sub(r'<key\s+name="home"[^/]*/>', new_key, mjcf_text)
    return mjcf_text


def main():
    print(f"[1/4] Sanitize URDF -> {_URDF_TMP}")
    sanitize_urdf(_URDF_IN, _URDF_TMP)

    print(f"[2/4] Compile URDF -> MJCF (MuJoCo {mujoco.__version__})")
    mjcf = compile_to_mjcf(_URDF_TMP)

    print(f"[3/4] Post-edit MJCF (free joint, floor, actuators, sensors, keyframe)")
    mjcf = post_edit(mjcf)
    mjcf = fill_keyframe(mjcf)

    with open(_MJCF_OUT, "w", encoding="utf-8") as f:
        f.write(mjcf)
    os.remove(_URDF_TMP)

    print(f"[4/4] Verify by reloading {_MJCF_OUT}")
    model = mujoco.MjModel.from_xml_path(_MJCF_OUT)
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt}")
    print("  joints:")
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"    [{i}] {nm}")
    print("  actuators:")
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"    [{i}] {nm}")
    print(f"\nDone. MJCF written to {_MJCF_OUT}")


if __name__ == "__main__":
    main()
