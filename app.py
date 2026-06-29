import gradio as gr

from pdf_loader import load_pdf
from retriever import answer


with gr.Blocks(title="Hindi Agriculture RAG Assistant") as demo:

    gr.Markdown("# 🌾 Hindi Agriculture Knowledge Assistant")

    gr.Markdown(
        """
Ask agriculture-related questions in Hindi.

You can optionally upload an agriculture PDF to extend the knowledge base.
"""
    )

    with gr.Row():

        # -------------------------
        # Left Panel
        # -------------------------

        with gr.Column(scale=1):

            pdf_input = gr.File(
                label="Upload Agriculture PDF (Optional)",
                file_types=[".pdf"]
            )

            pdf_status = gr.Textbox(
                label="PDF Status",
                value="No PDF loaded. Using built-in knowledge base.",
                interactive=False
            )

            pdf_input.change(
                fn=load_pdf,
                inputs=pdf_input,
                outputs=pdf_status
            )

        # -------------------------
        # Right Panel
        # -------------------------

        with gr.Column(scale=2):

            model_choice = gr.Radio(
                choices=[
                    "Local Qwen",
                    "Gemini API"
                ],
                value="Local Qwen",
                label="Choose LLM"
            )

            question_input = gr.Textbox(
                label="Ask your question in Hindi",
                placeholder="Example: यूरिया में कितना नाइट्रोजन होता है?",
                lines=2
            )

            ask_button = gr.Button(
                "Get Answer",
                variant="primary"
            )

            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False
            )

            ask_button.click(
                fn=answer,
                inputs=[
                    question_input,
                    model_choice
                ],
                outputs=answer_output
            )


demo.launch()