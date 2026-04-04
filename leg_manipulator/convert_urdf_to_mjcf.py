"""Convert go2_with_diy_leg URDF to MuJoCo XML for sim2sim."""

import os
import re
import shutil
import glob
import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "urdf", "go2_with_diy_leg.urdf")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "mjcf", "go2_diyleg.xml")

# Default joint angles from training config
DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.543,
    "FL_calf_joint": -2.07,
    "FL_diy_joint1": 0.1,
    "FL_diy_joint2": 0.0,
    "FL_diy_joint3": 0.0,
    "FL_diy_joint4": 0.0,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.543,
    "FR_calf_joint": -2.07,
    "diy_joint1": 0.1,
    "diy_joint2": 0.0,
    "diy_joint3": 0.0,
    "diy_joint4": 0.0,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.905,
    "RL_calf_joint": -1.78,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.905,
    "RR_calf_joint": -1.78,
}


def main():
    with open(URDF_PATH) as f:
        urdf_text = f.read()

    # Remove dont_collapse attribute
    urdf_text = urdf_text.replace(' dont_collapse="true"', '')

    # Collect all mesh files into one flat directory
    mesh_collect_dir = os.path.join(SCRIPT_DIR, "mjcf", "meshes")
    os.makedirs(mesh_collect_dir, exist_ok=True)
    for pattern in ["go2_description/dae/*.dae", "meshes/*.STL"]:
        for src in glob.glob(os.path.join(SCRIPT_DIR, pattern)):
            dst = os.path.join(mesh_collect_dir, os.path.basename(src))
            shutil.copy2(src, dst)

    # Strip package:// prefix, keep only basename, and convert .dae -> .obj
    def fix_mesh_filename(m):
        basename = m.group(1)
        if basename.endswith('.dae'):
            basename = basename[:-4] + '.obj'
        return f'filename="{basename}"'
    urdf_text = re.sub(
        r'filename="package://leg_manipulator/[^"]*?([^/"]+)"',
        fix_mesh_filename,
        urdf_text,
    )

    # Keep only the first <material> in each <visual> block
    mat_pattern = r'<material\s[^>]*(?:/>|>.*?</material>)'
    def replace_in_visual(match):
        block = match.group(0)
        mats = list(re.finditer(mat_pattern, block))
        if len(mats) <= 1:
            return block
        for m in reversed(mats[1:]):
            block = block[:m.start()] + block[m.end():]
        return block
    urdf_text = re.sub(r'<visual>.*?</visual>', replace_in_visual, urdf_text, flags=re.DOTALL)

    # Remove Gazebo-specific sections
    urdf_text = re.sub(r'<ros2_control[\s\S]*?</ros2_control>', '', urdf_text)
    urdf_text = re.sub(r'<gazebo[\s\S]*?</gazebo>', '', urdf_text)

    # Add MuJoCo compiler hint
    urdf_text = urdf_text.replace(
        '</robot>',
        '  <mujoco>\n'
        '    <compiler meshdir="' + mesh_collect_dir + '" balanceinertia="true" discardvisual="false"/>\n'
        '  </mujoco>\n'
        '</robot>'
    )

    # Write temp URDF
    os.makedirs(os.path.join(SCRIPT_DIR, "mjcf"), exist_ok=True)
    tmp_urdf = os.path.join(SCRIPT_DIR, "mjcf", "_tmp_mujoco.urdf")
    with open(tmp_urdf, "w") as f:
        f.write(urdf_text)

    # Step 1: Compile URDF with MuJoCo to get the model
    model = mujoco.MjModel.from_xml_path(tmp_urdf)
    print(f"Compiled URDF: nq={model.nq}, nv={model.nv}, njnt={model.njnt}")

    # Save intermediate XML
    mujoco.mj_saveLastXML(OUTPUT_PATH, model)

    # Step 2: Read and post-process the XML
    with open(OUTPUT_PATH) as f:
        xml_text = f.read()

    # Extract content between <worldbody> and </worldbody>
    wb_match = re.search(r'<worldbody>\s*(.*?)\s*</worldbody>', xml_text, re.DOTALL)
    wb_content = wb_match.group(1)

    # The base link's geoms are directly in worldbody (not wrapped in a body).
    # Split: base geoms (lines starting with <geom not inside a <body>) and child bodies.
    lines = wb_content.split('\n')
    base_geoms = []
    child_bodies = []
    in_body = False
    body_depth = 0
    body_lines = []

    for line in lines:
        stripped = line.strip()
        if not in_body:
            if stripped.startswith('<body '):
                in_body = True
                body_depth = 1
                body_lines = [line]
            elif stripped.startswith('<geom') or stripped.startswith('<inertial'):
                base_geoms.append(line)
        else:
            body_lines.append(line)
            if '<body ' in stripped and not '/>' in stripped:
                body_depth += 1
            if '</body>' in stripped:
                body_depth -= 1
                if body_depth == 0:
                    child_bodies.append('\n'.join(body_lines))
                    in_body = False
                    body_lines = []

    # Build the new worldbody with a floating base body
    base_inertial = (
        '      <inertial pos="0.021112 0 -0.005366" mass="6.921" '
        'diaginertia="0.107 0.098077 0.02448"/>'
    )

    new_worldbody = '  <worldbody>\n'
    new_worldbody += '    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n'
    new_worldbody += '    <geom name="ground" type="plane" size="50 50 0.1" rgba="0.8 0.85 0.8 1" condim="3" contype="1" conaffinity="1"/>\n'
    new_worldbody += '    <body name="base" pos="0 0 0.35">\n'
    new_worldbody += '      <freejoint name="root"/>\n'
    new_worldbody += f'{base_inertial}\n'
    new_worldbody += '      <site name="imu" pos="-0.02557 0 0.04232" size="0.01"/>\n'
    # Add base collision geoms
    for g in base_geoms:
        new_worldbody += '  ' + g + '\n'
    # Add child bodies
    for body in child_bodies:
        # Indent by 2 extra spaces
        indented = '\n'.join('    ' + l for l in body.split('\n'))
        new_worldbody += indented + '\n'
    new_worldbody += '    </body>\n'
    new_worldbody += '  </worldbody>'

    # Replace worldbody
    xml_text = re.sub(
        r'<worldbody>.*?</worldbody>',
        new_worldbody,
        xml_text,
        flags=re.DOTALL,
    )

    # Replace compiler with our settings
    xml_text = re.sub(
        r'<compiler[^/]*/>\s*',
        '<compiler angle="radian" meshdir="' + mesh_collect_dir + '" autolimits="true"/>\n'
        '  <option timestep="0.004" iterations="4" solver="Newton"/>\n'
        '  <default>\n'
        '    <geom friction="0.8 0.02 0.01" condim="3" contype="1" conaffinity="1"/>\n'
        '    <joint damping="0.5" armature="0.01"/>\n'
        '  </default>\n\n',
        xml_text,
        count=1,
    )

    # Add actuators for all joints
    actuator_lines = []
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name is None:
            continue
        if "diy" in jnt_name:
            kp, kd = (100.0, 5.0) if "joint1" in jnt_name else (10.0, 0.5)
        else:
            kp, kd = 25.0, 0.5
        actuator_lines.append(
            f'    <position name="{jnt_name}_actuator" joint="{jnt_name}" '
            f'kp="{kp}" kv="{kd}" ctrlrange="-100 100"/>'
        )
    actuator_block = "  <actuator>\n" + "\n".join(actuator_lines) + "\n  </actuator>\n"

    # Add sensors
    sensor_block = (
        "  <sensor>\n"
        '    <gyro name="gyro" site="imu"/>\n'
        '    <accelerometer name="accelerometer" site="imu"/>\n'
        '    <framequat name="imu_quat" objtype="site" objname="imu"/>\n'
        "  </sensor>\n"
    )

    # Build keyframe (now with freejoint: qpos has 7 + 20 = 27 entries)
    # freejoint: [x, y, z, qw, qx, qy, qz]
    # Then joint order from the original compiled model
    nq_new = 7 + model.nq  # freejoint + original joints
    qpos = np.zeros(nq_new)
    qpos[2] = 0.35  # z height
    qpos[3] = 1.0   # quat w
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name in DEFAULT_JOINT_ANGLES:
            qpos_adr = 7 + model.jnt_qposadr[i]  # offset by freejoint
            qpos[qpos_adr] = DEFAULT_JOINT_ANGLES[jnt_name]
    qpos_str = " ".join(f"{v:.6g}" for v in qpos)
    keyframe_block = (
        "  <keyframe>\n"
        f'    <key name="home" qpos="{qpos_str}"/>\n'
        "  </keyframe>\n"
    )

    # Insert all blocks before </mujoco>
    xml_text = xml_text.replace(
        "</mujoco>",
        "\n" + actuator_block + "\n" + sensor_block + "\n" + keyframe_block + "</mujoco>"
    )

    with open(OUTPUT_PATH, "w") as f:
        f.write(xml_text)

    # Step 3: Verify the final model
    model2 = mujoco.MjModel.from_xml_path(OUTPUT_PATH)
    data2 = mujoco.MjData(model2)
    mujoco.mj_resetDataKeyframe(model2, data2, 0)

    print(f"\nFinal model: nq={model2.nq}, nv={model2.nv}, nu={model2.nu}, nsensor={model2.nsensor}")

    print("\nJoint mapping (qpos_adr includes freejoint offset):")
    for i in range(model2.njnt):
        name = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_adr = model2.jnt_qposadr[i]
        qvel_adr = model2.jnt_dofadr[i]
        print(f"  [{i}] {name}  qpos={qpos_adr}  qvel={qvel_adr}")

    print("\nActuators:")
    for i in range(model2.nu):
        name = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}")

    print("\nSensors:")
    for i in range(model2.nsensor):
        name = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_SENSOR, i)
        print(f"  [{i}] {name}")

    # Clean up temp file
    os.remove(tmp_urdf)

    print(f"\nDone! MJCF saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
