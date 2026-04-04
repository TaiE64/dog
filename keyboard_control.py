"""Keyboard controller for Go2 sim2sim — run in a separate terminal.

Usage:
    python keyboard_control.py

Keys:
    W/S   : forward / backward
    A/D   : strafe left / right
    Q/E   : turn left / right
    Space : stop
    Ctrl+C: quit
"""

import socket
import sys
import termios
import tty

UDP_IP = "127.0.0.1"
UDP_PORT = 9871

VX_MAX = 1.0
VY_MAX = 0.4
VYAW_MAX = 1.0
VEL_STEP = 0.1


def get_key():
    """Read a single keypress (non-blocking-ish)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    vx, vy, vyaw = 0.0, 0.0, 0.0

    print("=" * 40)
    print("  Keyboard Control for Go2")
    print("=" * 40)
    print("  W/S   : forward / backward")
    print("  A/D   : strafe left / right")
    print("  Q/E   : turn left / right")
    print("  Space : stop")
    print("  Ctrl+C: quit")
    print("=" * 40)
    print(f"  vx={vx:.1f}  vy={vy:.1f}  yaw={vyaw:.1f}")

    try:
        while True:
            key = get_key()
            if key == '\x03':  # Ctrl+C
                break
            elif key in ('w', 'W'):
                vx = min(vx + VEL_STEP, VX_MAX)
            elif key in ('s', 'S'):
                vx = max(vx - VEL_STEP, -VX_MAX)
            elif key in ('a', 'A'):
                vy = min(vy + VEL_STEP, VY_MAX)
            elif key in ('d', 'D'):
                vy = max(vy - VEL_STEP, -VY_MAX)
            elif key in ('q', 'Q'):
                vyaw = min(vyaw + VEL_STEP, VYAW_MAX)
            elif key in ('e', 'E'):
                vyaw = max(vyaw - VEL_STEP, -VYAW_MAX)
            elif key == ' ':
                vx, vy, vyaw = 0.0, 0.0, 0.0

            msg = f"{vx:.2f} {vy:.2f} {vyaw:.2f}"
            sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
            print(f"\r  vx={vx:.1f}  vy={vy:.1f}  yaw={vyaw:.1f}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.sendto(b"0.00 0.00 0.00", (UDP_IP, UDP_PORT))
        sock.close()
        print("\nBye!")


if __name__ == "__main__":
    main()
