#!/usr/bin/env python3
"""
Keyboard control for walking + dual DIY arms.

Walking:
    i: forward    k: stop    ,: backward
    j: turn left  l: turn right
    u: fwd+left   o: fwd+right
    w/x: increase/decrease linear speed
    e/c: increase/decrease angular speed

Arm (active arm):
    1/2: joint1 (prismatic) extend/retract
    3/4: joint2 (revolute) +/-
    5/6: joint3 (revolute) +/-
    7/8: joint4 (gripper) open/close
    TAB: switch between FL/FR arm
    0: both grippers close
    9: both grippers open

    q: quit
"""
import sys
import tty
import termios
import rospy
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist

# FL arm
FL_JOINTS = {
    'FL_diy_joint1': {'min': 0.0, 'max': 0.1, 'step': 0.01, 'pos': 0.1},
    'FL_diy_joint2': {'min': -2.4, 'max': 0.67, 'step': 0.1, 'pos': 0.0},
    'FL_diy_joint3': {'min': -1.42, 'max': 1.45, 'step': 0.1, 'pos': 0.0},
    'FL_diy_joint4': {'min': -0.17, 'max': 1.4, 'step': 0.1, 'pos': -0.17},
}

# FR arm
FR_JOINTS = {
    'diy_joint1': {'min': 0.0, 'max': 0.1, 'step': 0.01, 'pos': 0.1},
    'diy_joint2': {'min': -2.4, 'max': 0.67, 'step': 0.1, 'pos': 0.0},
    'diy_joint3': {'min': -1.42, 'max': 1.45, 'step': 0.1, 'pos': 0.0},
    'diy_joint4': {'min': -0.17, 'max': 1.4, 'step': 0.1, 'pos': -0.17},
}

ARMS = {'FL': FL_JOINTS, 'FR': FR_JOINTS}
ARM_JOINT_ORDER = ['joint1', 'joint2', 'joint3', 'joint4']

ARM_KEY_MAP = {
    '1': (0, +1),  # joint1 +
    '2': (0, -1),  # joint1 -
    '3': (1, +1),  # joint2 +
    '4': (1, -1),  # joint2 -
    '5': (2, +1),  # joint3 +
    '6': (2, -1),  # joint3 -
    '7': (3, +1),  # joint4 +
    '8': (3, -1),  # joint4 -
}

# Walking velocity
WALK_KEYS = {
    'i': (1, 0, 0),   # forward
    ',': (-1, 0, 0),  # backward
    'j': (0, 0, 1),   # turn left
    'l': (0, 0, -1),  # turn right
    'u': (1, 0, 1),   # fwd + left
    'o': (1, 0, -1),  # fwd + right
    'k': (0, 0, 0),   # stop
}


def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main():
    rospy.init_node('arm_keyboard_control')

    # Arm publishers for both arms
    arm_pubs = {}
    for joints in ARMS.values():
        for name in joints:
            arm_pubs[name] = rospy.Publisher(f'/{name}_controller/command', Float64, queue_size=1)

    # Walk publisher
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    # Tell walk_controller to stop publishing DIY targets
    ext_pub = rospy.Publisher('/diy_external_control', Bool, queue_size=1, latch=True)
    rospy.sleep(0.5)
    ext_pub.publish(Bool(data=True))

    speed = 0.5
    turn = 1.0
    vx, vz = 0.0, 0.0
    active_arm = 'FL'

    print("=" * 60)
    print("  Walk + Dual Arm Keyboard Control")
    print("=" * 60)
    print("  Walk: i=fwd  k=stop  ,=back  j/l=turn  u/o=diag")
    print("        w/x=speed up/down  e/c=turn up/down")
    print("  Arm:  1/2=joint1  3/4=joint2  5/6=joint3")
    print("        7/8=gripper open/close")
    print("        TAB=switch FL/FR arm")
    print("        9=both open  0=both close")
    print("  q: quit")
    print("=" * 60)

    try:
        while not rospy.is_shutdown():
            key = get_key()

            if key == 'q' or key == '\x03':
                break

            # TAB: switch active arm
            if key == '\t':
                active_arm = 'FR' if active_arm == 'FL' else 'FL'

            # Arm control (active arm)
            if key in ARM_KEY_MAP:
                joint_idx, direction = ARM_KEY_MAP[key]
                joints = ARMS[active_arm]
                jname = list(joints.keys())[joint_idx]
                j = joints[jname]
                j['pos'] += direction * j['step']
                j['pos'] = max(j['min'], min(j['max'], j['pos']))

            # 9: both grippers open, 0: both close
            if key == '9':
                FL_JOINTS['FL_diy_joint4']['pos'] = 1.4
                FR_JOINTS['diy_joint4']['pos'] = 1.4
            elif key == '0':
                FL_JOINTS['FL_diy_joint4']['pos'] = -0.17
                FR_JOINTS['diy_joint4']['pos'] = -0.17

            # Walk control
            if key in WALK_KEYS:
                vx = WALK_KEYS[key][0] * speed
                vz = WALK_KEYS[key][2] * turn
            elif key == 'w':
                speed = min(speed + 0.1, 2.0)
            elif key == 'x':
                speed = max(speed - 0.1, 0.0)
            elif key == 'e':
                turn = min(turn + 0.1, 3.0)
            elif key == 'c':
                turn = max(turn - 0.1, 0.0)

            # Publish both arms
            for joints in ARMS.values():
                for name, j in joints.items():
                    arm_pubs[name].publish(Float64(j['pos']))

            # Publish walk
            twist = Twist()
            twist.linear.x = vx
            twist.angular.z = vz
            cmd_pub.publish(twist)

            # Print state
            aj = ARMS[active_arm]
            jvals = list(aj.values())
            print(f"\r  [{active_arm}] vel=({vx:+.1f},{vz:+.1f}) spd={speed:.1f} trn={turn:.1f} "
                  f"j1={jvals[0]['pos']:+.3f} "
                  f"j2={jvals[1]['pos']:+.2f} "
                  f"j3={jvals[2]['pos']:+.2f} "
                  f"j4={jvals[3]['pos']:+.2f}   ", end="", flush=True)

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cmd_pub.publish(Twist())
        ext_pub.publish(Bool(data=False))
        print("\nBye!")


if __name__ == '__main__':
    main()
