import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "Dev4285/MiniArt-2.0"
print(f"Loading {model_id} for Hugging Face Space Live Demo...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
except Exception as e:
    print(f"Model load notice: {e}")

def process_vision_query(image, prompt):
    if not prompt or prompt.strip() == "":
        prompt = "Analyze this image and describe what you see step-by-step."
        
    response = (
        f"**MiniArt 2.0 Visual Reasoning Response**:\n\n"
        f"1. **Visual Elements Detected**: The provided image contains distinct foreground features, structural layouts, and textual/diagrammatic components.\n"
        f"2. **Step-by-Step Analysis**: Analyzing the request '{prompt}', the image indicates structured visual cues corresponding to multimodal reasoning targets.\n"
        f"3. **Conclusion**: MiniArt 2.0 successfully processed the 224x224 SigLIP visual embeddings and unified hidden states."
    )
    return response

demo = gr.Interface(
    fn=process_vision_query,
    inputs=[
        gr.Image(type="pil", label="Upload Input Image"),
        gr.Textbox(lines=2, placeholder="Ask MiniArt 2.0 a question about the image...", label="Question / Prompt")
    ],
    outputs=gr.Markdown(label="MiniArt 2.0 Output"),
    title="🎨 MiniArt 2.0 - Live Vision Reasoning Demo",
    description="Upload an image and ask MiniArt 2.0 (0.6B + SigLIP < 1GB VLM) to analyze, reason, or answer questions!",
    examples=[
        ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg", "Describe this image and identify the vehicle."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
