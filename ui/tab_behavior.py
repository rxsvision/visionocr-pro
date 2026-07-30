"""行为分析 Tab (P2 可选) — 即将推出"""
import gradio as gr


def create_tab_behavior(config: dict, registry):
    gr.Markdown(
        """
### 行为分析模块 — 即将推出

本模块正在开发中，计划集成以下能力：

| 能力 | 方案 | 状态 |
|------|------|------|
| 人体姿态估计 | YOLO-Pose (30 FPS, RTX 4070 Ti) | 规划中 |
| 违规行为判定 | 规则引擎 (确定性 OK/NG) | 规划中 |
| 事件存档 | VLM 自然语言描述 (疑似违规补充) | 规划中 |

**设计原则**：YOLO-Pose + 规则引擎为主（确定性、低延迟），VLM 仅补充自然语言存档。

启用条件：`config.yaml` → `behavior.enabled: true`（当前版本无实际效果）
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 视频流（预览）")
            video_input = gr.Video(label="摄像头 / 视频文件", interactive=False)
            start_btn = gr.Button("开始分析", variant="primary", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("#### 实时状态（预览）")
            pose_display = gr.Image(label="姿态叠加", interactive=False)
            with gr.Row():
                action_box = gr.Textbox(label="当前动作", interactive=False)
                fatigue_box = gr.Textbox(label="疲劳指数", interactive=False)
            alert_box = gr.Textbox(
                label="报警信息", lines=3, interactive=False,
                value="模块开发中，敬请期待。",
            )

