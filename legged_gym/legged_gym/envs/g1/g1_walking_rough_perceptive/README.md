# G1 仿真地形感知崎岖路面行走

本目录在 `g1_walking_rough` 的基础上增加仿真局部高度感知，但不修改原崎岖路面任务、正在运行的训练进程或原日志。新任务名称为：

```text
g1_walking_rough_perceptive
```

当前项目形成三组可直接对照的实验：

| 实验 | Actor 输入 | 用途 |
|---|---:|---|
| Blind Baseline | 141维本体历史 | 当前正在训练的 `g1_walking_rough` |
| Oracle Perception | 141维本体历史 + 63维精确高度 | 测量地形感知能够达到的性能上限 |
| Noisy Perception | 141维本体历史 + 63维退化高度 | 模拟可部署传感器，测试噪声鲁棒性 |

这里的“部署”仅指仿真部署。当前没有真实深度相机和真实G1，因此不能据此宣称已经完成真实机器人Sim-to-Real。

## 1. 为什么不需要等待当前训练结束

代码开发、形状检查和PPO链路验证都可以提前完成。当前盲走策略训练结束后，只需通过环境变量指定它的checkpoint，新任务即可从该策略开始训练：

```text
正在训练的 g1_walking_rough
          │
          │ 复制原Actor、std和形状兼容的Critic
          ▼
g1_walking_rough_perceptive
          │
          ├── Oracle高度图训练
          └── Noisy高度图训练
```

感知Actor采用零初始化的残差接口。刚完成权重迁移时，地形分支对动作的贡献严格为零，因此新Actor的动作与原141维盲走Actor完全一致。训练开始后，PPO再逐渐学习何时利用地形信息。这避免了随机初始化的地形网络立刻破坏已经学会的步态。

## 2. 观测构造

### 2.1 141维本体历史

单帧本体观测保持原walking policy的47维布局：

| 部分 | 维度 | 含义 |
|---|---:|---|
| `command_obs` | 3 | 机身坐标系中的前向、侧向和偏航速度命令 |
| `base_ang_vel` | 3 | 机身角速度 |
| `projected_gravity` | 3 | 重力在机身坐标系中的投影，用于表征姿态 |
| `dof_pos-default` | 12 | 12个受控关节相对默认姿态的角度 |
| `dof_vel` | 12 | 12个受控关节角速度 |
| `previous_action` | 12 | 上一个策略周期输出的关节动作 |
| `sin_phase, cos_phase` | 2 | 周期步态相位编码 |

连续堆叠三帧：

\[
o_t^{\mathrm{prop}}
=
[o_{t-2}^{47},o_{t-1}^{47},o_t^{47}]
\in\mathbb R^{141}.
\]

### 2.2 63维局部高度图

高度采样点与盲走任务的Critic保持一致：

\[
x\in\{-0.8,-0.6,\ldots,0.8\},\qquad
y\in\{-0.6,-0.4,\ldots,0.6\}.
\]

共计：

\[
N_h=9\times7=63.
\]

对第 \(i\) 个采样点，仿真中的真实高度残差定义为：

\[
h_{t,i}
=
\operatorname{clip}
\left(
z_t^{\mathrm{base}}
-z^{\mathrm{target}}
-z_{t,i}^{\mathrm{terrain}},
-0.5,0.5
\right).
\]

其中：

- \(z_t^{\mathrm{base}}\)：机器人基座世界高度；
- \(z^{\mathrm{target}}=0.78\,\mathrm m\)：期望机身离地高度；
- \(z_{t,i}^{\mathrm{terrain}}\)：随机器人平移和偏航后的第 \(i\) 个地形采样高度；
- \(h_{t,i}>0\)：该点地形相对较低；
- \(h_{t,i}<0\)：该点地形相对较高。

最后乘以原项目的高度观测缩放系数5.0，再输入Actor。

### 2.3 Actor和Critic总维度

Actor输入为：

\[
o_t^{A}=[o_t^{\mathrm{prop}},\tilde h_t]
\in\mathbb R^{141+63}
=\mathbb R^{204}.
\]

Critic仍接收当前50维特权本体状态与精确63维高度：

\[
o_t^{C}\in\mathbb R^{50+63}=\mathbb R^{113}.
\]

这仍然是非对称Actor–Critic：Actor面对退化感知，Critic在训练期间使用精确信息辅助价值估计。

## 3. 仿真传感器退化模型

Oracle模式直接使用 \(h_t\)。Noisy模式使用：

\[
\tilde h_t
=
\operatorname{clip}
\left(
M_t\odot
Q(h_{t-d}+b+n_t)
 +(1-M_t)h_{\mathrm{fill}},
-h_{\max},h_{\max}
\right).
\]

变量含义：

| 变量 | 当前设置 | 含义 |
|---|---:|---|
| \(d\) | 0～5个策略周期 | 每个环境、每回合随机采样的感知延迟 |
| \(b\) | ±1.5 cm | 对整张高度图一致的相关高度偏置 |
| \(n_t\) | 标准差0～2 cm | 每个环境、每回合随机强度的逐点高斯噪声 |
| \(M_t\) | 92%有效 | 每个高度点是否有效的随机掩码 |
| \(h_{\mathrm{fill}}\) | 0 m | 丢失点的填充值 |
| \(Q(\cdot)\) | 5 mm | 高度量化分辨率 |
| \(h_{\max}\) | 0.5 m | 高度观测截断范围 |

当前策略周期约为20 ms，因此5个周期对应约100 ms的最大感知延迟。

所有传感器参数位于 `g1_walking_rough_perceptive_config.py` 的 `perception` 类中，可以单独做噪声、延迟和丢点消融实验。

## 4. 地形编码器和残差融合

地形分支结构为：

\[
63\rightarrow128\rightarrow32.
\]

得到地形隐变量：

\[
z_t^{\mathrm{terrain}}=E_\theta(\tilde h_t)
\in\mathbb R^{32}.
\]

原盲走Actor的第一层为：

\[
p_t=W_po_t^{\mathrm{prop}}+b_p
\in\mathbb R^{512}.
\]

新Actor使用残差融合：

\[
f_t=p_t+W_z z_t^{\mathrm{terrain}}+b_z.
\]

随后保持原Actor的后两层：

\[
f_t\rightarrow512\rightarrow128\rightarrow12.
\]

初始化时：

\[
W_z=0,\qquad b_z=0,
\]

所以任意地形输入都满足：

\[
\pi_{\mathrm{perceptive}}(o^{\mathrm{prop}},\tilde h)
=
\pi_{\mathrm{blind}}(o^{\mathrm{prop}}).
\]

## 5. 与正在训练的盲走策略衔接

建议先让当前 `g1_walking_rough` 训练到性能基本稳定，不要求必须等到10000次迭代。选择checkpoint时应同时观察：

- 机器人跌倒率是否明显下降；
- 地形等级是否持续上升；
- 速度跟踪奖励是否稳定；
- 是否存在只在低等级地形有效的情况；
- 带窗口播放时是否出现严重脚滑或拖脚。

感知任务能够迁移以下内容：

- 盲走Actor的3个线性层，共6个权重/偏置张量；
- 12维动作标准差 `std`；
- 当输入形状一致时，迁移113维崎岖Critic；
- PPO优化器重新初始化；
- 地形编码器和地形残差层重新初始化。

如果当前训练尚未结束，可以先用中间checkpoint验证训练流程；正式对比时，再统一换成选定的Baseline checkpoint重新开始Oracle和Noisy实验。

## 6. 训练命令

首先准备环境：

```bash
conda activate safe_track
export TORCH_EXTENSIONS_DIR=/tmp/safe_track_torch_extensions
export MPLCONFIGDIR=/tmp/safe_track_matplotlib
export PYTHONPATH=legged_gym:rsl_rl
export WANDB_MODE=offline
```

### 6.1 Oracle实验

```bash
export G1_PERCEPTIVE_WARM_START=/绝对路径/model_xxx.pt
export G1_PERCEPTIVE_MODE=oracle

python -m legged_gym.envs.g1.g1_walking_rough_perceptive.train_perceptive \
  --task g1_walking_rough_perceptive \
  --run_name oracle_heightmap_34 \
  --entity local \
  --wandb g1_walking_rough_perceptive \
  --headless
```

### 6.2 Noisy实验

Oracle和Noisy实验应从同一个盲走checkpoint开始，才能形成公平对照：

```bash
export G1_PERCEPTIVE_WARM_START=/绝对路径/model_xxx.pt
export G1_PERCEPTIVE_MODE=noisy

python -m legged_gym.envs.g1.g1_walking_rough_perceptive.train_perceptive \
  --task g1_walking_rough_perceptive \
  --run_name noisy_heightmap_34 \
  --entity local \
  --wandb g1_walking_rough_perceptive \
  --headless
```

可选环境变量：

| 环境变量 | 作用 |
|---|---|
| `G1_PERCEPTIVE_NUM_ENVS` | 覆盖并行环境数 |
| `G1_PERCEPTIVE_MAX_ITERATIONS` | 覆盖最大训练迭代数 |
| `G1_PERCEPTIVE_WARM_START` | 指定盲走崎岖策略checkpoint |
| `G1_PERCEPTIVE_MODE` | `oracle`或`noisy` |
| `G1_PERCEPTIVE_TRANSFER_CRITIC=0` | 不迁移Critic |
| `G1_PERCEPTIVE_DISABLE_WARM_START=1` | 完全随机初始化，仅用于消融 |

训练日志独立保存在：

```text
legged_gym/logs/g1_walking_rough_perceptive/<run_name>/
```

不会写入原来的 `g1_walking_rough` 日志目录。

## 7. 验证和冒烟训练

CPU PhysX观测与网络验证：

```bash
TORCH_EXTENSIONS_DIR=/tmp/safe_track_torch_extensions \
MPLCONFIGDIR=/tmp/safe_track_matplotlib \
PYTHONPATH=legged_gym:rsl_rl \
conda run -n safe_track python -m \
legged_gym.envs.g1.g1_walking_rough_perceptive.validate_perceptive
```

运行一次完整的PPO采样、更新与checkpoint保存：

```bash
TORCH_EXTENSIONS_DIR=/tmp/safe_track_torch_extensions \
MPLCONFIGDIR=/tmp/safe_track_matplotlib \
PYTHONPATH=legged_gym:rsl_rl \
conda run -n safe_track python -m \
legged_gym.envs.g1.g1_walking_rough_perceptive.smoke_train_perceptive
```

当前已验证：

- Actor观测形状为 `(N, 204)`；
- Critic观测形状为 `(N, 113)`；
- 动作形状为 `(N, 12)`；
- Oracle高度误差为0；
- 权重迁移后的最大动作误差为0；
- 一轮PPO后地形残差层参数发生更新；
- checkpoint能够正常保存。

## 8. Blind、Oracle和Noisy统一评测

评测脚本支持原盲走任务和新感知任务，输出JSON结果。推荐至少使用256个环境和多个完整episode。

评测Blind Baseline：

```bash
export G1_EVAL_TASK=g1_walking_rough
export G1_EVAL_CHECKPOINT=/绝对路径/blind_model_xxx.pt
export G1_EVAL_NUM_ENVS=256
export G1_EVAL_STEPS=5000

python -m legged_gym.envs.g1.g1_walking_rough_perceptive.evaluate_policy \
  --task g1_walking_rough --headless
```

评测Oracle或Noisy策略：

```bash
export G1_EVAL_TASK=g1_walking_rough_perceptive
export G1_EVAL_CHECKPOINT=/绝对路径/perceptive_model_xxx.pt
export G1_PERCEPTIVE_MODE=noisy

python -m legged_gym.envs.g1.g1_walking_rough_perceptive.evaluate_policy \
  --task g1_walking_rough_perceptive --headless
```

当前统一输出指标包括：

- 跌倒次数；
- 正常episode超时次数；
- 平均平面速度跟踪误差；
- 平均机身姿态误差；
- 支撑脚平均滑移速度；
- 平均关节力矩平方；
- 感知策略额外输出平均高度图误差。

正式实验应固定随机种子、环境数、测试步数、推力设置和测试地形范围，并至少重复3个随机种子，报告均值与标准差。

## 9. 文件说明

| 文件 | 作用 |
|---|---|
| `terrain_perception.py` | 批量高度传感器退化模型 |
| `g1_walking_rough_perceptive_config.py` | 观测、传感器和PPO配置 |
| `g1_walking_rough_perceptive.py` | 204维Actor观测和113维Critic观测构造 |
| `train_perceptive.py` | 独立训练与盲走权重迁移 |
| `validate_perceptive.py` | CPU PhysX环境和网络形状验证 |
| `smoke_train_perceptive.py` | 单次完整PPO链路测试 |
| `evaluate_policy.py` | Blind与Perceptive统一指标评测 |
| `actor_critic_perceptive.py` | 位于`rsl_rl/modules`的残差地形编码Actor–Critic |

## 10. 当前能力边界

本版本实现的是高度场感知，不是完整深度相机渲染。它适合验证“机器人提前知道附近地形高度是否有用”，并能高效支持大量并行环境。

当前尚未包含：

- 深度图到点云或高程图的重建；
- 遮挡、反光和材质相关的相机误差；
- 机器人位姿漂移导致的地图坐标误差；
- Teacher–Student特征蒸馏；
- Isaac Gym到MuJoCo的感知策略迁移；
- 真实机器人和真实传感器部署。

后续应先完成Blind、Oracle、Noisy三组实验。如果Oracle相对Blind没有明显提升，应该先检查地形采样范围、奖励和地形难度，而不是立即增加更复杂的相机模型。如果Oracle有效但Noisy明显退化，再针对延迟、噪声、缺失点分别做消融并调整鲁棒训练范围。

