# MiniMax H3 统一调用协议

Mac ComfyUI Desktop 统一使用 **Save (API Format)** 导出的 workflow，通过同一个
RunPod Serverless endpoint 提交。Worker 不会把 Director、Motion Context 或
ControlNet 的节点改写成 I2V；提交的 workflow 是最终图，Worker 只负责上传素材、
绑定同类 loader，并注入通用的 prompt、尺寸、帧数、采样器和 seed。

## 支持模式

`params.mode` 可取：

* `t2v`：纯文本
* `i2v`：单张首帧图片
* `fl2v`：首帧/尾帧工作流（由 Desktop workflow 提供对应 loader）
* `r2v`：图片、视频、音频参考
* `v2v`：视频参考/视频编辑
* `rv2v`：视频加图片或音频参考

功能模式需要对应的 ComfyUI workflow 和模型节点。当前 MXFP8 I2V 模板仍然
保持原样；R2V/V2V/RV2V 应使用 Ref2VA workflow，因为它们不是 FL2VA 的同一
推理图。

## 请求格式

```json
{
  "input": {
    "workflow": {"...": "ComfyUI Save (API Format) JSON"},
    "params": {
      "mode": "r2v", "prompt": "...", "negative_prompt": "",
      "width": 1344, "height": 768, "duration": 15, "fps": 24,
      "steps": 8, "seed": 123456, "sampler": "res_multistep",
      "scheduler": "simple", "cfg": 1.0, "lora_strength": 1.0
    },
    "assets": [
      {"type": "image", "name": "character.png", "data": "<base64>"},
      {"type": "video", "name": "motion.mp4", "data": "<base64>"},
      {"type": "audio", "name": "voice.wav", "data": "<base64>"}
    ]
  }
}
```

`images` 仍然兼容旧版 I2V 请求。也可以使用 `references.images`、
`references.videos`、`references.audios`；统一协议会把它们合并后按 workflow
中 loader 的顺序绑定。超过 workflow 中 loader 数量的素材会被拒绝前请先在
Desktop 中增加对应的 loader 节点，避免素材被静默丢弃。

## 推荐的功能节点包

这些能力应作为独立的“功能完整”镜像/Endpoint 安装，不覆盖三个 MXFP8 性能
Endpoint：

1. `AIMixer/ComfyUI_MiniMaxH3_Director`：分镜、时间线、T2V/I2V/FL2V/R2V/V2V/RV2V。
2. `NikoDemon80/ComfyUI-H3-Motion-Context`：跨片段动作和音频续接。
3. H3 Ref2VA workflow：图片、视频、图片+视频、音频参考。
4. H3 Fun ControlNet Union：姿态、深度、边缘和控制视频；先作为实验能力验证。

H3 没有稳定的官方“绘制一条 spline 后人物严格沿线走”接口。人物轨迹目前
应使用姿态/深度控制视频配合 ControlNet，或使用 Motion Context 续接；Director
负责按时间线安排镜头和人物事件。

## Mac 调用规则

不要手动拼接 ComfyUI 节点 JSON：在 Mac Desktop 打开对应模板，选择
**Save (API Format)**，将整个 JSON 放入 `input.workflow`，再把素材转成
`input.assets`。这样普通 I2V 和功能完整的 Director workflow 共用同一 HTTP
入口，缺少节点时会返回明确错误，而不是生成错误视频。

