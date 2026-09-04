# H3 功能接入矩阵

| 能力 | 当前三个性能 Endpoint | 功能完整 Endpoint |
|---|---:|---:|
| I2V | 已支持 | 已支持 |
| T2V | 需提交 T2V workflow | 已规划 |
| FL2V | 需提交 FL2V workflow | 已规划 |
| R2V | 需 Ref2VA workflow | 已规划 |
| V2V / RV2V | 需 Ref2VA workflow | 已规划 |
| 多图/多视频/音频参考 | Worker API 已支持上传绑定 | 需对应 workflow loader |
| Director 分镜时间线 | 未内置 | Feature Easy 镜像集成 H3 Director |
| Motion Context 长视频续接 | 未内置 | H3 Director 可选能力，需对应工作流 |
| 姿态/深度/轨迹控制 | 未内置 | ControlNet 实验节点 |
| MXFP8 / Native Attention / SageAttention | 性能 Endpoint 已隔离测试 | 不默认强制 |

性能 Endpoint 保持单一、可复现的 H3 图；功能节点包、Ref2VA 和 ControlNet 会
改变模型图或显存峰值，不应和速度基准混在一起。

H3 Director 仅安装在 Feature Easy 镜像中。它把 T2V/I2V/FL2V 与 R2V/V2V/RV2V
集中到一张画布，并由任务类型选择 FL2VA 或 Ref2VA 分支；性能 Endpoint 不加载
该节点包。
