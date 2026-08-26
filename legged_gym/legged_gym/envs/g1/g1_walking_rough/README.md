# G1 崎岖地形 Walking Policy

这个任务在现有 `g1_walking_unitree` 基础上训练中等崎岖路面的 G1 行走策略。任务名为：

```text
g1_walking_rough
```

## 1. 设计目标

第一版采用“本体感知 actor + 地形特权 critic”的非对称训练：

```text
actor:  原 walking policy 的 3 x 47 = 141 维观测
critic: 当前 50 维机器人状态 + 9 x 7 = 63 维局部高度图 = 113 维
action: 12 个腿关节动作
```

actor 不读取高度图，因此训练完成的网络仍能使用现有 walking 部署接口，不需要在真机上立即增加深度相机或局部高程图。代价是它适合连续起伏、斜坡、中低台阶和离散矮障碍，不适合需要提前选择落脚点的深坑、宽沟或稀疏踏脚石。

## 2. 复用现有策略的方式

默认从下面的平地 checkpoint 开始：

```text
legged_gym/logs/g1_walking_unitree/seed_31_1/model_20000.pt
```

平地和崎岖地形 actor 结构完全相同：

```text
141 -> 512 -> ELU -> 128 -> ELU -> 12
```

训练入口只迁移：

- actor 的 6 个权重/偏置张量；
- 12 维动作标准差 `std`。

critic 输入从 50 维变成 113 维，所以 critic 和 PPO optimizer 从头初始化，不能直接恢复平地 optimizer。

## 3. 地形课程

地图由 10 个难度行和 10 个地形类型列组成。机器人初始只分配到难度 0–1；单回合行走距离超过半个地形块时升级，明显低于命令要求时降级。

地形类型比例为：

| 地形 | 比例 |
|---|---:|
| 平地 | 10% |
| 随机连续起伏 | 20% |
| 上坡 | 15% |
| 下坡 | 15% |
| 上台阶 | 15% |
| 下台阶 | 15% |
| 离散矮障碍 | 10% |

设课程难度为 $d\in[0,0.9]$，主要参数为：

$$
h_{\mathrm{rough}}=0.01+0.045d,
$$

$$
s_{\mathrm{slope}}=0.04+0.20d,
$$

$$
h_{\mathrm{step}}=0.025+0.11d,
$$

$$
h_{\mathrm{obstacle}}=0.02+0.09d.
$$

最高课程附近对应约 ±5 cm 随机起伏、0.22 坡度、12.4 cm 台阶和 10.1 cm 离散障碍。第一版有意不生成深坑和沟壑。

## 4. 观测

### 4.1 Actor：仍为 141 维

单帧 47 维与原 walking policy 一致：

| 范围 | 维数 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | 机身速度命令 |
| `[3:6]` | 3 | 基座角速度 |
| `[6:9]` | 3 | 投影重力 |
| `[9:21]` | 12 | 相对默认关节角 |
| `[21:33]` | 12 | 关节速度 |
| `[33:45]` | 12 | 上一动作 |
| `[45:47]` | 2 | 步态相位 sin/cos |

三帧按旧到新拼接为 141 维，因此可以复用平地 actor 和现有 walking 推理代码。

### 4.2 Critic：113 维

critic 当前帧包含：

```text
3 command
+ 3 base linear velocity
+ 3 base angular velocity
+ 3 projected gravity
+ 12 joint position
+ 12 joint velocity
+ 12 previous action
+ 2 gait phase
+ 63 terrain heights
= 113
```

63 个高度采样点位于机身水平面附近的 $[-0.8,0.8]\times[-0.6,0.6]$ m 区域。输入的是“相对期望基座高度”的地形高度，而不是世界绝对高度。

## 5. 崎岖地形奖励修正

原平地任务中的绝对高度奖励不能直接用于崎岖地形。本任务改为：

$$
e_h=(z_{\mathrm{base}}-z_{\mathrm{terrain,base}})-h_{\mathrm{target}}.
$$

摆脚高度同样使用足端位置减去该足端正下方的地形高度。另增加：

- `feet_slip`：接触脚的水平速度惩罚；
- `feet_stumble`：水平接触力远大于竖直接触力时惩罚；
- `feet_contact_forces`：超过 350 N 的足端冲击惩罚；
- 更平滑的动作变化和较温和的竖直速度惩罚。

这些修改避免机器人在斜坡或台阶上为了追求世界绝对高度而错误下蹲或抬升。

## 6. 域随机化

默认随机化：

- 摩擦系数：`[0.45, 1.25]`；
- 基座附加质量：`[-1, 2] kg`；
- 每 7 秒左右的水平速度冲击，最大 `0.55 m/s`；
- 关节、角速度和重力方向观测噪声。

这些范围比原平地任务初期的推力更温和，防止课程地形、宽速度命令和强推力同时出现导致 warm-start gait 被迅速破坏。

## 7. 环境验证

`safe_track` 首次导入 Isaac Gym 时需要可写的 Torch 扩展目录：

```bash
conda activate safe_track
export TORCH_EXTENSIONS_DIR=/tmp/safe_track_torch_extensions
export MPLCONFIGDIR=/tmp/safe_track_matplotlib
export PYTHONPATH=legged_gym:rsl_rl
```

创建 8 个 CPU 环境并检查 141/113 维观测：

```bash
python -m legged_gym.envs.g1.g1_walking_rough.validate_rough
```

执行一次包含 warm-start、采样、PPO 更新和 checkpoint 保存的最小测试：

```bash
python -m legged_gym.envs.g1.g1_walking_rough.smoke_train_rough
```

当前验证结果：

```text
rough env validation passed: obs=(8, 141) critic=(8, 113)
rough PPO smoke training passed
actor_parameter_change=0.00097489
```

## 8. 正式训练

GPU 机器上从项目根目录运行：

```bash
conda activate safe_track
export TORCH_EXTENSIONS_DIR=/tmp/safe_track_torch_extensions
export MPLCONFIGDIR=/tmp/safe_track_matplotlib
export PYTHONPATH=legged_gym:rsl_rl
export WANDB_MODE=offline

python -m legged_gym.envs.g1.g1_walking_rough.train_rough \
  --task g1_walking_rough \
  --run_name rough_warmstart_33 \
  --entity local \
  --wandb g1_walking_rough \
  --headless
```

默认使用 4096 个并行环境、10000 次 PPO iteration，每 500 次保存一次。日志位于：

```text
legged_gym/logs/g1_walking_rough/rough_warmstart_33/
```

可用环境变量做短训练或换 checkpoint：

```bash
export G1_ROUGH_NUM_ENVS=1024
export G1_ROUGH_MAX_ITERATIONS=2000
export G1_ROUGH_WARM_START=/absolute/path/to/model_20000.pt
```

完全从头训练：

```bash
export G1_ROUGH_DISABLE_WARM_START=1
```

在线使用 Weights & Biases 时把 `WANDB_MODE` 改为 `online`，并把 `--entity` 改成自己的账号或团队。

## 9. 查看训练结果

现有 `play.py` 已把 `g1_walking_rough` 识别为 walking 任务。进入 `legged_gym` 目录后运行：

```bash
cd legged_gym

python -m legged_gym.scripts.play \
  --task g1_walking_rough \
  --experiment_name g1_walking_rough \
  --load_run rough_warmstart_33 \
  --checkpoint -1
```

播放时关闭观测噪声、推力和摩擦随机化，但保留随机崎岖地形，以便观察策略本身的地形通过能力。

## 10. 训练阶段建议

不要只看总 reward，至少同时监控：

- `terrain_level`：课程难度是否持续上升；
- `tracking_lin_vel`、`tracking_ang_vel`：速度跟踪是否退化；
- `orientation`、`base_height`：姿态和离地高度；
- `feet_slip`、`feet_stumble`、`feet_contact_forces`：落脚质量；
- episode length 和 dones：是否频繁摔倒。

推荐按三个阶段判断 checkpoint：

1. 0–1000 iteration：保持平地 gait，低级起伏不频繁摔倒；
2. 1000–4000 iteration：课程等级上升，斜坡和低台阶跟踪稳定；
3. 4000 iteration 以后：扩大速度分布，检查各类地形成功率和 sim-to-sim 泛化。

如果平均课程等级长期停留在 0，优先降低 `max_push_vel_xy`、初始速度范围或最高初始地形等级，不要第一步就继续增加网络规模。

## 11. 查看指定checkpoint的带窗口效果

独立播放入口只加载策略推理，不会修改checkpoint、训练进程或训练日志。导出仓库默认播放最终的 `model_10000.pt`，在第5级随机起伏地面上以0.35 m/s向前行走：

```bash
conda activate safe_track
export TORCH_EXTENSIONS_DIR=/tmp/safe_track_torch_extensions
export MPLCONFIGDIR=/tmp/safe_track_matplotlib
export PYTHONPATH=legged_gym:rsl_rl

python -m legged_gym.envs.g1.g1_walking_rough.play_rough_checkpoint \
  --task g1_walking_rough
```

可以通过环境变量切换测试条件：

```bash
export G1_ROUGH_PLAY_CHECKPOINT=legged_gym/logs/g1_walking_rough/rough_warmstart_33/model_10000.pt
export G1_ROUGH_PLAY_TERRAIN=random_rough
export G1_ROUGH_PLAY_LEVEL=5
export G1_ROUGH_PLAY_VX=0.35
```

`G1_ROUGH_PLAY_TERRAIN` 可选：`flat`、`random_rough`、`slope_up`、`slope_down`、`stairs_up`、`stairs_down`、`obstacles`。难度等级范围为0～9。窗口中按 `ESC` 退出，按 `V` 暂停或恢复渲染。

默认持续运行到关闭窗口。设置 `G1_ROUGH_PLAY_DURATION=30` 可在30秒后自动结束；`G1_ROUGH_PLAY_HEADLESS=1` 仅用于无窗口诊断。

最高难度全地形巡检可以使用：

```bash
export G1_ROUGH_SUITE_CHECKPOINT=legged_gym/logs/g1_walking_rough/rough_warmstart_33/model_10000.pt
export G1_ROUGH_SUITE_LEVEL=9
export G1_ROUGH_SUITE_REPLICAS=1

python -m legged_gym.envs.g1.g1_walking_rough.play_rough_suite \
  --task g1_walking_rough \
  --sim_device cuda:0 \
  --rl_device cuda:0
```

摄像机会依次切换随机起伏、上下坡、上下台阶和离散凸起。最终 `model_10000.pt` 的完整 Isaac Gym 与 MuJoCo 记录见 `deploy/rough_walking_mujoco/MODEL_10000_VALIDATION.md`。
