# walk_and_grasp 控制流程

## 顶层结构

`mujoco.set_mjcb_control(ctrl.get_control)` 把 `get_control` 注册为 MuJoCo 物理 step 的回调。`mj_step` 每 `timestep=2ms` 调用一次，但 `get_control` 内部计数器只让逻辑每 `n_substeps=10` 步（即 **20ms / 50Hz**）真正执行，其余直接 return。所以 *policy_dt = 20ms*，但 *physics_dt = 2ms*。

每次执行做 7 件事：vision → state machine 转换 → cmd_vel → 选 j4 → walk policy 推理 → 选 arm pose → slew + 写 actuator。

---

## 1. Vision（10Hz，比 control loop 慢一半）

[`Vision.detect`](walk_and_grasp.py#L222) 流程：

1. 渲染 `head_cam` RGB
2. HSV 阈值得 red mask（`r>150 & g<80 & b<80`）
3. 算 bbox。判断长轴：`bbox_w >= bbox_h` 说明 bar 横躺在图像里，反之则竖立
4. **找 thin section**：`_longest_thin_run` 沿长轴扫 thickness（每列/行的红色像素数），threshold = midpoint，取最长连续段。run 中点 = thin slice 位置
5. **取 silhouette 两端**：thin slice 那一列（或行）的红色像素 first/last index = bar 投影的两个对侧表面点
6. 渲染 depth，对这两个像素 deproject 到世界系，**世界中点 = bar 中轴**（几何上精确，与相机角度无关）
7. 返回 `goal_world` + `partial`（bbox 触图像边即 partial，不可信）

## 2. Goal locking ([get_control 第①块](walk_and_grasp.py#L781))

防 close-range 视觉抖动：

- 前 5 次完整检测累积进 `_goal_lock_buf`，平均后 lock
- Locked 后忽略 partial detection
- Locked + 在 SEARCH/APPROACH 阶段时，若 drift > 0.15m → 认为 cube 移动，重新 lock
- 其他阶段（PRE_ALIGN/EXTEND/SERVO/CLOSE/LIFT_TEST）即使检测到 drift 也不 re-lock，避免 arm 自己撞 bar 时把目标改写

`vision_age` 看门狗：APPROACH 阶段且未 lock 时若 1.5s 没新检测 → REACQUIRE。

## 3. State Machine ([get_control](walk_and_grasp.py#L843))

```
                  ┌───────────────────────────────────────────────┐
SEARCH ──────────►│ APPROACH ── PRE_ALIGN ── EXTEND ── SERVO ──── │
  ▲                                                       │       │
  │                                                       ▼       ▼
  │ (4s timeout)                  ┌─── RETRY_BACKUP ◄── LIFT_TEST◄CLOSE
  └─── REACQUIRE ◄── (vision lost)│  (≤RETRY_MAX=3)         │
                                  │                         │ (held)
                                  └──────────────────► DONE ◄
```

| Phase | 退出条件 | cmd_vel | arm | gripper |
|---|---|---|---|---|
| SEARCH | `goal_world` 出现 | `(0,0,0.4)` 慢转扫描 | HOLD | OPEN |
| APPROACH | `‖base - approach_xy‖ < 0.08m` | 朝 approach_xy 走（forward + 0.3×lateral + yaw） | HOLD | OPEN |
| PRE_ALIGN | `yaw_err < 0.15rad` 或超时 8s | 只 yaw，不 forward | HOLD | OPEN |
| EXTEND | `phase_t > 1.0s` | 0 | ramp HOLD → reach_pose | OPEN |
| SERVO | `d_ee_to_goal < 0.05m` 或超时 6s | `_set_servo_command` 驱动 | IK 实时跟踪 goal_world | OPEN |
| CLOSE | `phase_t > 0.6s` | 0 | reach_pose | **CLOSE** |
| LIFT_TEST | `phase_t > 1.2s` (ramp 0.6 + hold 0.6) | 0 | ramp reach → lift_pose（EE +6cm） | **CLOSE** |
| RETRY_BACKUP | `phase_t > 1.0s` | `(-0.20, ±0.10, 0)` 后退+左右摆 | HOLD（slew 平滑过渡） | OPEN |
| DONE | — | 0 | HOLD | CLOSE if held else OPEN |
| REACQUIRE | vision_age<0.3s → APPROACH; 4s 超时 → SEARCH | `(-0.25,0,0)` 后退 | HOLD | OPEN |

### Approach pose 几何 ([_compute_approach_pose](walk_and_grasp.py#L634))

- `BAR_AXIS_XY = (0,1)` → perp 方向是 (±1, 0)
- 选 perp 让 dog 站在 bar 朝向自己的那一侧（`dot(base-goal, perp) > 0`）
- `approach_xy = goal[:2] + perp×0.40 + bar_axis×(-ee_offset_body[1])`
  - `0.40m` 是 perp 方向 standoff（避开 basket 0.075m 半宽 + FL_hip 0.193m + 5cm safety）
  - `bar_axis × -ee_offset[1]` 沿 bar 方向偏移，补偿 FL arm 天生左偏 8cm（用平移而不是 yaw 倾斜身体来对齐）
- 目标 yaw 让 body forward = -perp（直面 bar）

### Reach pose 计算 ([_compute_reach_pose](walk_and_grasp.py#L601))

PRE_ALIGN 完成时调用一次，用 **当前 live base pose** 解 IK：在 body 系 `(REACH_POSE_FORWARD=0.38, REACH_POSE_SIDE=0.08, target_z=goal_z)` 摆个标准伸展姿态，记录解出的 j1/j2/j3 和 EE 在 body 系的实际偏移 `_ee_offset_body`。

### SERVO 控制律 ([_set_servo_command](walk_and_grasp.py#L702))

不是简单"走到固定距离"，而是 body+arm 协同：

1. **目标 yaw**：让 body forward 延长后 ≈ EE → cube 方向（补偿 arm 左偏角）
2. **理想 body 位置**：`ideal_xy = cube - R(target_yaw) @ ee_offset_xy`
3. **forward 速度**：阶梯式
   - `d_body_to_cube < 0.32m`: 停（防 dog 鼻子撞）
   - `ik_err > 0.05`: 顶 `SERVO_MIN_VX=0.20` 推 body 前进（IK 还够不到）
   - `|body_forward_err| > 0.04`: 比例控制 fine-tune
   - else: 停
4. **lateral 始终 0**（quadruped 横向跟踪差），靠 yaw 把 forward 重定向
5. **arm IK** 同时跑 `warm_only=True` + `j1_max=0.02` + `_slew_reach_pose` 限速 → 平滑跟随 cube

### LIFT_TEST 验证 ([get_control](walk_and_grasp.py#L949))

CLOSE 进入 LIFT_TEST 前快照 `target_handle` 几何中心的 z，IK 解出"当前 EE 抬高 6cm"的 lift_pose。LIFT_TEST 期间 arm 从 reach_pose ramp 到 lift_pose（gripper 保持 CLOSE），1.2s 后比对：

- `dz = handle_z_now - handle_z_start > 0.02m` AND `‖handle - EE‖ < 0.13m` → 真夹住 → DONE
- 否则 retry_count++，超过 `RETRY_MAX=3` → DONE 失败，否则 RETRY_BACKUP

并行接触检测：每帧扫 `data.contact`，记录 cube body 与 FL arm body 是否曾接触（仅 logging 用）。

## 4. Walk policy（永远跑）

ONNX 推理 [`_build_obs`](walk_and_grasp.py#L770) → `_last_action`：obs 是 `[gyro×0.2, gravity, cmd_vel, joint_pos-DEFAULT, joint_vel×0.05, last_action]` (45 维)，输出 12 维 leg action。targets = `action × 0.25 + DEFAULT_OFFSETS`。

**关键**：腿全程由 policy 控制，状态机只输出 cmd_vel。所有阶段（包括 EXTEND/CLOSE/LIFT_TEST）都在跑 walk policy，dog 站立姿态由 cmd_vel=0 时的 policy 自然 stand 维持。

## 5. Arm targets ([get_control](walk_and_grasp.py#L1093))

j1 是 prismatic slider，独立处理：

- EXTEND: 与 j2/j3 同步 ramp
- SERVO/CLOSE: IK 直出
- LIFT_TEST: 用 lift_pose
- 其他: 以 0.03 m/s slew 回到 J1_RETRACTED=0.10

j2/j3：阶段直接选目标值（HOLD / IK / blend），然后 **统一 slew 限制 0.8 rad/s**（[get_control](walk_and_grasp.py#L1126)）—— 防止 LIFT_TEST→RETRY_BACKUP 等切换时瞬移。

j4 (gripper)：阶段直接选 OPEN/CLOSE，无 slew（夹爪要快速响应）。

## 6. 写 actuator

12 leg + 4 FL DIY + 4 FR DIY（FR 始终 hold 在 `[0.05,-1.06,0,J4_OPEN]`）。

---

## 注意点（已知非完美）

1. **APPROACH 用 lateral，SERVO 不用** — APPROACH 距离远容许 quadruped 横向漂移，SERVO 近距离精度不够所以靠 yaw + forward。
2. **视觉 goal 在 cylinder bar 上几何精确**，但对均匀 cylinder（无明显 thin section）算法会 degenerate；对当前 MJCF 的 dumbbell-style bar 工作良好，对纯 cube 不可用。
3. **basket.xml 里的 "target_cube" 实际是篮子**，红色把手是 `target_handle` geom（z 偏移 +0.068m）。所有 ground-truth 检查用 geom 不用 body。
4. **PRE_ALIGN 有 8s 超时** — walk policy 静站时 yaw 控制不精，超过容忍直接进 EXTEND。
5. 没有 vision-loss 看门狗覆盖 PRE_ALIGN/EXTEND/SERVO 之后阶段（设计如此 — 这些阶段距离近，相机本来就可能丢目标，靠 locked 的 world goal 工作）。
