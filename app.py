from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from agent.pipeline import VoiceAgentPipeline


ROOT = Path(__file__).resolve().parent
INTENTS_PATH = ROOT / "data" / "intents.json"

pipeline = VoiceAgentPipeline(workspace_root=ROOT, intents_path=INTENTS_PATH)


def format_trace(result: Dict[str, Any]) -> str:
    transcript = result["transcript"]
    intents = result["intents"]
    tool_results = result["tool_results"]

    trace = (
        "### Pipeline Trace\n"
        f"1. **Speech/Text Input**\n"
        f"   - Model: `{transcript['model']}`\n"
        f"   - Duration: `{transcript['duration_seconds']:.2f}s`\n"
        f"   - Transcript: {transcript['text']}\n\n"
        f"2. **Intent Classification**\n"
    )

    for i, intent in enumerate(intents):
        trace += (
            f"   - **Intent {i+1}**: `{intent['intent']}` "
            f"(Confidence: `{intent['confidence']:.3f}`)\n"
        )

    trace += f"\n3. **Tool Execution**\n"
    for i, tool in enumerate(tool_results):
        trace += (
            f"   - **Tool {i+1}**: `{tool['tool_name']}` (Success: `{tool['ok']}`)\n"
            f"   - Output: {tool['output']}\n"
        )
    return trace


def format_history() -> str:
    if not pipeline.history.get_history():
        return "No history yet."

    lines: List[str] = ["### Session History"]
    for i, entry in enumerate(pipeline.history.get_history()):
        transcript = entry["transcript"]["text"]
        intents = entry["intents"]
        intent_str = ", ".join(
            [f"{intent['intent']} ({intent['confidence']:.2f})" for intent in intents]
        )
        lines.append(f"{i+1}. **{transcript}** -> _{intent_str}_")
    return "\n".join(lines)


def _result_to_output(result: Dict[str, Any]) -> str:
    tool_results = result.get("tool_results", [])
    if not tool_results:
        return "No tool result. Try a different command."
    return "\n\n".join(item.get("output", "") for item in tool_results)


def process_audio(
    audio_path: str, chat_history: list[dict[str, str]]
) -> Tuple[str, list[dict[str, str]], str]:
    if not audio_path:
        return "", chat_history, "No audio input."

    result_obj = pipeline.run(audio_path=audio_path)
    result = result_obj.to_dict()
    trace = format_trace(result)
    history = format_history()

    response = _result_to_output(result)
    chat_history.extend(
        [
            {"role": "user", "content": result["transcript"]["text"]},
            {"role": "assistant", "content": response},
        ]
    )
    return history, chat_history, trace


def process_text(
    text: str, chat_history: list[dict[str, str]]
) -> Tuple[str, list[dict[str, str]], str]:
    if not text:
        return "", chat_history, "No text input."

    result_obj = pipeline.run(audio_path=None, manual_text=text)
    result = result_obj.to_dict()
    trace = format_trace(result)
    history = format_history()

    response = _result_to_output(result)
    chat_history.extend(
        [
            {"role": "user", "content": result["transcript"]["text"]},
            {"role": "assistant", "content": response},
        ]
    )
    return history, chat_history, trace


# Gradio App
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
).set(
    body_background_fill="#f0f4f8",
    body_text_color="#1f2937",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="*secondary_200",
    button_secondary_text_color="#1f2937",
    border_color_accent="#e5e7eb",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f3f4f6",
    shadow_drop="rgba(0, 0, 0, 0.05) 0px 1px 2px 0px",
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif;
    background: #f0f4f8;
    color: #1f2937;
}

.main-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08);
    padding: 2rem !important;
}

.panel {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 16px !important;
    box-shadow: none !important;
}

#chatbot {
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
    background: #f9fafb !important;
    box-shadow: none !important;
}

#chatbot .message {
    color: #1f2937 !important;
}

#chatbot .message.user {
    background: #e0f2fe !important;
}

#chatbot .message.assistant {
    background: #f1f5f9 !important;
}

#trace, #history {
    padding: 1.5rem;
}

#trace *, #history * {
    color: #374151 !important;
    text-decoration: none !important;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #111827;
}

.hero-sub {
    font-size: 1.1rem;
    color: #4b5563;
    margin-top: 0.5rem;
}

.gr-button {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.gr-input {
    border-radius: 8px !important;
}
"""

with gr.Blocks(title="Voice AI Agent") as app:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 1.6rem 0 1.2rem 0;">
            <h1 class="hero-title">Voice AI Agent</h1>
            <p class="hero-sub">
                A personal voice assistant for browsing and local tasks.
            </p>
        </div>
    """
    )

    with gr.Row(equal_height=True, elem_classes=["main-card"]):
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                height=550,
            )
            with gr.Group():
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="Type your command here...",
                        label="Text Input",
                        scale=4,
                    )
                    submit_btn = gr.Button("➤", variant="primary", scale=1, min_width=0)
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Voice Input",
                    )

        with gr.Column(scale=1):
            with gr.Accordion("Pipeline Trace", open=True, elem_classes=["panel"]):
                trace_output = gr.Markdown("Trace will appear here.", elem_id="trace")
            with gr.Accordion("Session History", open=True, elem_classes=["panel"]):
                history_output = gr.Markdown(
                    "History will appear here.", elem_id="history"
                )

    audio_input.stop_recording(
        process_audio,
        inputs=[audio_input, chatbot],
        outputs=[history_output, chatbot, trace_output],
    )
    submit_btn.click(
        process_text,
        inputs=[text_input, chatbot],
        outputs=[history_output, chatbot, trace_output],
    )
    text_input.submit(
        process_text,
        inputs=[text_input, chatbot],
        outputs=[history_output, chatbot, trace_output],
    )

if __name__ == "__main__":
    app.launch(debug=True, show_error=True, theme=theme, css=css)
