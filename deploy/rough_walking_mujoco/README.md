# `10000.pt` 崎岖地形策略的 MuJoCo 部署

该目录是独立部署模块，不修改训练任务、已有 walking 部署或 A* + MPC 代码。它直接读取
RSL-RL 的 `model_10000.pt`，复现训练时的 `3 × 47 = 141` 维 actor 观测，并用与原部署一致的
12 腿关节 PD 控制和 11 个上身关节锁定控制驱动 G1。

## 快速运行

在项目根目录执行。窗口模式推荐先激活环境再直接运行 Python；在当前机器上，这可避免
`conda run` 包装 GLFW 窗口时偶发的退出阶段段错误：

```bash
conda activate safe_track
MUJOCO_GL=glfw python -u -m deploy.rough_walking_mujoco.deploy_mujoco \
  --terrain random_rough --level 9 --duration 30
```

无窗口回归：

```bash
conda run --no-capture-output -n safe_track \
  python -m deploy.rough_walking_mujoco.deploy_mujoco \
  --terrain random_rough --level 9 --duration 10 --headless
```

可选地形为 `flat`、`random_rough`、`up_slope`、`down_slope`、`up_stairs`、
`down_stairs`、`discrete_obstacles` 和 `mixed`。`--level` 范围为 0–9；9 对应约
±5.05 cm 随机起伏、12.41° 坡度、12.4 cm 台阶和 10.1 cm 离散障碍。

默认前进命令是 `vx=0.25 m/s`。可以通过 `--vx`、`--vy` 和 `--yaw-rate` 修改。
部署默认读取：

```text
legged_gym/logs/g1_walking_rough/rough_warmstart_33/model_10000.pt
```

## 跨仿真说明

MuJoCo 与训练使用的 GPU PhysX 在接触求解、网格表达和摩擦模型上不同，因此这里的结果既是
效果展示，也是一次 sim-to-sim 泛化测试。MuJoCo 高度场在运行时生成，不会改写机器人 XML。
程序会输出摔倒次数、平面速度跟踪误差、最小机身离地高度、最大倾角和前进距离。

本次 `10000.pt` 的完整验证数据见 [MODEL_10000_VALIDATION.md](MODEL_10000_VALIDATION.md)。

## Noisy感知策略

Noisy感知策略直接读取：

```text
legged_gym/logs/g1_walking_rough_perceptive/noisy_heightmap_34/model_10000.pt
```

MuJoCo每个策略周期在机器人周围发射9×7条向下射线，生成与训练一致的63维局部高度图。
窗口中的绿色球是有效采样点，红色球是传感器退化模型随机丢失的采样点。

```bash
conda activate safe_track
MUJOCO_GL=glfw python -u -m deploy.rough_walking_mujoco.deploy_perceptive_mujoco \
  --terrain mixed --level 9 --duration 15 \
  --perception-mode noisy --sensor-seed 33
```

`sensor-seed=33`对应本次压力测试所用的100 ms最大感知延迟。感知MuJoCo验证数据见
[PERCEPTIVE_MODEL_10000_VALIDATION.md](PERCEPTIVE_MODEL_10000_VALIDATION.md)。
