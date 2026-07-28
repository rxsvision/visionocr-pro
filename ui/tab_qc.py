"""工业质检 Tab"""
import gradio as gr


def create_tab_qc(config: dict, registry):
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 图像输入")
            qc_image = gr.Image(label="拍照 / 上传", type="filepath")
            with gr.Row():
                camera_btn = gr.Button("相机采集")
                run_qc_btn = gr.Button("执行检测", variant="primary")

            gr.Markdown("### 检测配置")
            detect_mode = gr.Radio(
                choices=["缺陷检测 (Dinomaly)", "开放词汇 (Grounding DINO)",
                         "通用检测 (RF-DETR)", "条码读取"],
                value="缺陷检测 (Dinomaly)",
                label="检测模式",
            )
            prompt_input = gr.Textbox(
                label="缺陷提示词 (开放词汇模式)",
                placeholder="划痕.凹陷.色差.毛刺.",
                visible=True,
            )

        with gr.Column(scale=2):
            gr.Markdown("### 检测结果")
            result_image = gr.Image(label="标注结果")
            with gr.Row():
                verdict_box = gr.Textbox(label="判定", scale=1)
                score_box = gr.Textbox(label="异常分数", scale=1)
            barcode_box = gr.Textbox(label="条码内容", lines=2)
            detail_json = gr.JSON(label="详细结果")

    run_qc_btn.click(
        fn=_run_qc,
        inputs=[qc_image, detect_mode, prompt_input],
        outputs=[result_image, verdict_box, score_box, barcode_box, detail_json],
    )


def _run_qc(image_path, mode, prompt):
    if not image_path:
        return None, "—", "—", "—", {"error": "请先上传图片或采集"}
    # TODO: Phase 4 实现
    return (
        None,
        "NG (占位)",
        "0.87",
        "QR: RXV-2026-0001 (占位)",
        {"mode": mode, "prompt": prompt, "status": "stub"},
    )
