import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def run_inference(image_path=None, prompt="Explain the reasoning behind this step-by-step."):
    model_id = "Dev4285/MiniArt-2.0"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    result = run_inference(prompt="What is 15 * 14?")
    print(result)
