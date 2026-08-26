# Oracle `model_10000.pt` Play 验证

验证 checkpoint：`logs/g1_walking_rough_perceptive/oracle_heightmap_34/model_10000.pt`。

验证条件：GPU PhysX、`level=9`、`vx=0.25 m/s`、每类8个机器人、10秒、评测种子33。
评测种子与 Blind `model_10000.pt` 验证一致，因此使用相同的程序化地形布局。

| 地形 | Oracle无摔倒率 | Blind无摔倒率 | Oracle速度误差 | Blind速度误差 | Oracle姿态误差 | Blind姿态误差 |
|---|---:|---:|---:|---:|---:|---:|
| 平地 | 100% | 100% | 0.080 | 0.077 | 0.028 | 0.038 |
| 随机崎岖 | 100% | 100% | 0.100 | 0.104 | 0.035 | 0.043 |
| 下坡 | 100% | 100% | 0.099 | 0.101 | 0.033 | 0.044 |
| 上坡 | 100% | 100% | 0.083 | 0.084 | 0.028 | 0.039 |
| 下台阶 | 100% | 100% | 0.087 | 0.088 | 0.021 | 0.042 |
| 上台阶 | 100% | 100% | 0.112 | 0.121 | 0.037 | 0.047 |
| 离散障碍 | 100% | 87.5% | 0.163 | 0.123 | 0.055 | 0.048 |

Oracle在所有地形上的高度图平均误差均为0。除离散障碍外，其崎岖、坡面和台阶上的速度误差与
姿态误差整体低于Blind。离散障碍的无摔倒率从87.5%提高至100%，但10秒平均前进距离从
1.998 m降至1.568 m，说明策略通过更保守的运动换取安全性。

窗口版以单机器人在六类非平地场景中依次切换镜头，连续播放24秒，各场景均未发生摔倒。

复现命令：

```bash
G1_PERCEPTIVE_SUITE_HEADLESS=0 \
G1_PERCEPTIVE_SUITE_MODE=oracle \
G1_PERCEPTIVE_SUITE_LEVEL=9 \
G1_PERCEPTIVE_SUITE_CHECKPOINT=legged_gym/logs/g1_walking_rough_perceptive/oracle_heightmap_34/model_10000.pt \
PYTHONPATH=legged_gym:rsl_rl \
python -m legged_gym.envs.g1.g1_walking_rough_perceptive.play_perceptive_suite \
  --sim_device cuda:0 --rl_device cuda:0
```
