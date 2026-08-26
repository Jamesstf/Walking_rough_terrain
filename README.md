# Walking Rough Terrain

面向 Unitree G1 的崎岖地形行走项目，包含从平地策略热启动、盲走崎岖地形训练、Oracle/Noisy 局部高度感知训练，到 MuJoCo sim-to-sim 部署验证的完整流程。仓库只保留这条技术链所需的代码、机器人资源、最终 checkpoint 和验证记录，不包含原工程中的导航、搬运和轨迹跟踪模块。

## 项目能力

- 在 GPU PhysX / Isaac Gym 中训练 G1 通过随机起伏、上下坡、上下台阶和离散矮障碍。
- Blind Actor 不读取地形，只依赖三帧本体观测；Critic 使用精确局部高度图进行非对称训练。
- Perceptive Actor 增加 `9 × 7` 局部高度图，并提供 Oracle 与 Noisy 两种感知模式。
- Noisy 模式模拟延迟、高斯噪声、整图偏置、随机丢点和量化误差。
- 在 MuJoCo 中复现 12 腿关节 PD 控制、局部高度 raycast 和最高难度地形，验证 sim-to-sim 泛化。

## 完整流程

```text
平地 walking model_20000.pt
        │ 仅迁移 Actor，Critic 和优化器重新初始化
        ▼
Blind rough model_10000.pt
        │ 迁移本体 Actor；新增零初始化地形残差分支
        ├───────────────┐
        ▼               ▼
Oracle height map   Noisy height map
model_10000.pt      model_10000.pt
        │               │
        └──── Isaac Gym play / MuJoCo sim-to-sim ────► 指标与可视化
```

训练地形采用 10 级 curriculum。Level 9 约对应 `±5.05 cm` 随机起伏、`12.41°` 坡度、`12.4 cm` 台阶和 `10.1 cm` 离散障碍。训练同时随机化摩擦系数 `[0.45, 1.25]`、基座附加质量 `[-1, 2] kg`，并施加最大约 `0.55 m/s` 的水平扰动。

## 观测与网络

Blind Actor 的单帧观测为 47 维：

| 索引 | 维数 | 含义 |
|---|---:|---|
| `[0:3]` | 3 | 前向、侧向和偏航速度命令 |
| `[3:6]` | 3 | 基座角速度 |
| `[6:9]` | 3 | 重力在机身坐标系中的投影，用于表示姿态 |
| `[9:21]` | 12 | 受控关节相对默认姿态的角度 |
| `[21:33]` | 12 | 受控关节角速度 |
| `[33:45]` | 12 | 上一个策略动作 |
| `[45:47]` | 2 | 步态相位的 `sin/cos` 编码 |

连续三帧按旧到新堆叠，得到 `3 × 47 = 141` 维本体历史。Blind Actor 为 `141 → 512 → 128 → 12`；Blind Critic 使用当前 50 维机器人状态与 63 维精确高度，共 113 维。

Perceptive Actor 输入为：

\[
o_t^A=[o_t^{prop},\tilde h_t]\in\mathbb R^{141+63}=\mathbb R^{204}.
\]

63 个采样点覆盖机器人附近 `[-0.8, 0.8] × [-0.6, 0.6] m` 的 `9 × 7` 网格。Oracle 直接使用精确高度；Noisy 模式采用：

\[
\tilde h_t=\operatorname{clip}\left(M_t\odot Q(h_{t-d}+b+n_t)+(1-M_t)h_{fill},-h_{max},h_{max}\right).
\]

| 变量 | 设置 | 含义 |
|---|---:|---|
| `d` | 0–5 个策略周期 | 随机感知延迟，最大约 100 ms |
| `b` | ±1.5 cm | 整张高度图共享的相关偏置 |
| `n_t` | 标准差 0–2 cm | 逐点高斯噪声 |
| `M_t` | 92% 有效 | 随机丢点掩码 |
| `h_fill` | 0 m | 丢失点填充值 |
| `Q` | 5 mm | 高度量化分辨率 |
| `h_max` | 0.5 m | 高度截断范围 |

更详细的奖励、观测和训练设计见 [Blind 任务说明](legged_gym/legged_gym/envs/g1/g1_walking_rough/README.md) 与 [感知任务说明](legged_gym/legged_gym/envs/g1/g1_walking_rough_perceptive/README.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `legged_gym/.../g1_walking_unitree` | 平地任务与热启动网络定义 |
| `legged_gym/.../g1_walking_rough` | Blind 崎岖地形环境、训练、play 与评测 |
| `legged_gym/.../g1_walking_rough_perceptive` | Oracle/Noisy 感知环境、训练与评测 |
| `rsl_rl/rsl_rl/modules/actor_critic_perceptive.py` | 地形编码器与残差融合 Actor |
| `deploy/rough_walking_mujoco` | Blind/Noisy 策略的 MuJoCo 部署 |
| `legged_gym/resources/robots/g1` | G1 URDF、MJCF 与 mesh |
| `legged_gym/logs` | 通过 Git LFS 保存的 4 个必要 checkpoint |

## 环境配置

本项目已在 Ubuntu、Python 3.8.19、PyTorch 2.0.1+cu118、NumPy 1.20.0、MuJoCo 3.2.3 和 NVIDIA Isaac Gym Preview 4 下验证。Isaac Gym 不能随本仓库分发，需要从 NVIDIA 单独获取并安装。

```bash
conda create -n safe_track python=3.8 -y
conda activate safe_track

# 根据本机 CUDA 安装匹配的 PyTorch；已验证组合为 torch 2.0.1+cu118。
pip install -r requirements.txt
pip install -e rsl_rl
pip install -e legged_gym

export TORCH_EXTENSIONS_DIR=/tmp/safe_track_torch_extensions
export MPLCONFIGDIR=/tmp/safe_track_matplotlib
```

克隆仓库时需要拉取 LFS 文件：

```bash
git lfs install
git lfs pull
```

## 训练

在仓库根目录运行。默认使用离线 WandB；可通过 `WANDB_MODE=online` 改为在线记录。

Blind 崎岖地形训练：

```bash
python -m legged_gym.envs.g1.g1_walking_rough.train_rough \
  --task g1_walking_rough --headless \
  --run_name rough_warmstart_33 --wandb g1_walking_rough
```

Oracle 感知训练：

```bash
G1_PERCEPTIVE_MODE=oracle \
python -m legged_gym.envs.g1.g1_walking_rough_perceptive.train_perceptive \
  --task g1_walking_rough_perceptive --headless \
  --run_name oracle_heightmap_34 --wandb g1_walking_rough_perceptive
```

Noisy 感知训练：

```bash
G1_PERCEPTIVE_MODE=noisy \
python -m legged_gym.envs.g1.g1_walking_rough_perceptive.train_perceptive \
  --task g1_walking_rough_perceptive --headless \
  --run_name noisy_heightmap_34 --wandb g1_walking_rough_perceptive
```

## Isaac Gym 播放

Blind 策略最高难度全地形测试：

```bash
G1_ROUGH_SUITE_LEVEL=9 \
python -m legged_gym.envs.g1.g1_walking_rough.play_rough_suite \
  --task g1_walking_rough --sim_device cuda:0
```

感知策略全地形测试，默认读取 Oracle checkpoint；可用 `G1_PERCEPTIVE_SUITE_CHECKPOINT` 指向 Noisy checkpoint：

```bash
G1_PERCEPTIVE_SUITE_LEVEL=9 \
G1_PERCEPTIVE_SUITE_CHECKPOINT=legged_gym/logs/g1_walking_rough_perceptive/noisy_heightmap_34/model_10000.pt \
python -m legged_gym.envs.g1.g1_walking_rough_perceptive.play_perceptive_suite \
  --task g1_walking_rough_perceptive --sim_device cuda:0
```

## MuJoCo 部署

窗口运行 Noisy 感知策略：

```bash
MUJOCO_GL=glfw python -u -m deploy.rough_walking_mujoco.deploy_perceptive_mujoco \
  --terrain mixed --level 9 --duration 15 \
  --perception-mode noisy --sensor-seed 33
```

无窗口回归：

```bash
python -m deploy.rough_walking_mujoco.deploy_perceptive_mujoco \
  --terrain up_stairs --level 9 --duration 10 \
  --perception-mode noisy --headless
```

可选地形包括 `flat`、`random_rough`、`up_slope`、`down_slope`、`up_stairs`、`down_stairs`、`discrete_obstacles` 和 `mixed`。

## 已验证结果

Noisy `model_10000.pt` 在 MuJoCo Level 9、`0.25 m/s` 命令和最严格 100 ms 延迟压力测试中的结果：

| 地形 | 时长 | 摔倒 | 速度误差 | 高度图误差 | 最大倾角 | 前进距离 |
|---|---:|---:|---:|---:|---:|---:|
| 随机崎岖 | 10 s | 0 | 0.089 m/s | 1.54 cm | 3.46° | 2.17 m |
| 上坡 | 10 s | 0 | 0.077 m/s | 1.83 cm | 3.30° | 2.05 m |
| 下坡 | 10 s | 0 | 0.075 m/s | 1.81 cm | 3.30° | 2.43 m |
| 上台阶 | 10 s | 0 | 0.138 m/s | 2.10 cm | 4.30° | 1.46 m |
| 下台阶 | 10 s | 0 | 0.087 m/s | 2.18 cm | 3.30° | 2.22 m |
| 离散障碍 | 10 s | 0 | 0.075 m/s | 1.75 cm | 3.30° | 2.22 m |
| 混合长地形 | 15 s | 0 | 0.081 m/s | 1.80 cm | 3.74° | 3.13 m |

完整记录见 [Blind model_10000 验证](deploy/rough_walking_mujoco/MODEL_10000_VALIDATION.md) 与 [Noisy model_10000 验证](deploy/rough_walking_mujoco/PERCEPTIVE_MODEL_10000_VALIDATION.md)。MuJoCo 与训练使用的 GPU PhysX 在接触与摩擦模型上不同，因此这些结果属于仿真到仿真的部署验证，不代表真机安全保证。

## Checkpoint

| 用途 | 路径 |
|---|---|
| 平地 Actor 热启动 | `legged_gym/logs/g1_walking_unitree/seed_31_1/model_20000.pt` |
| Blind 崎岖策略 | `legged_gym/logs/g1_walking_rough/rough_warmstart_33/model_10000.pt` |
| Oracle 感知策略 | `legged_gym/logs/g1_walking_rough_perceptive/oracle_heightmap_34/model_10000.pt` |
| Noisy 感知策略 | `legged_gym/logs/g1_walking_rough_perceptive/noisy_heightmap_34/model_10000.pt` |

## 许可与来源

代码基于 legged_gym、RSL-RL、Unitree G1 资源和 NVIDIA Isaac Gym 生态构建。各上游代码、模型与机器人资源保留各自许可；详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本仓库未包含 Isaac Gym SDK，也尚未对真实机器人部署进行安全认证。
