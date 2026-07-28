"""行为分析 Tab (P2 可选)"""
import gradio as gr


def create_tab_behavior(config: dict, registry):
    enabled = config.get("behavior", {}).get("enabled", False)

    if not enabled:
        gr.Markdown(
            "### 行为分析模块 (P2 可选)\n\n"
            "当前未启用。在 `config.yaml` 中设置 `behavior.enabled: true` 后重启。"
        )
        return

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 视频流")
            video_input = gr.Video(label="摄像头 / 视频文件")
            start_btn = gr.Button("开始分析", variant="primary")
            stop_btn = gr.Button("停止")

        with gr.Column(scale=2):
            gr.Markdown("### 实时状态")
            pose_display = gr.Image(label="姿态叠加")
            with gr.Row():
                action_box = gr.Textbox(label="当前动作")
                fatigue_box = gr.Textbox(label="疲劳指数")
            alert_box = gr.Textbox(label="报警信息", lines=3)
            event_log = gr.Dataframe(
                headers=["时间", "事件", "置信度"],
                label="事件日志",
            )

    start_btn.click(fn=_start_analysis, outputs=[alert_box])
    stop_btn.click(fn=lambda: "已停止", outputs=[alert_box])


def _start_analysis():
    # TODO: Phase 6 实现
    return "行为分析引擎尚未接入 (Phase 6)。将集成 RTMPose + CTR-GCN。"
