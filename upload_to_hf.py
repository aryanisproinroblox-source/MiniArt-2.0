import os
import sys

def main():
    print("=" * 65)
    print("HUGGING FACE MODEL RELEASE - Dev4285/MiniArt-2.0")
    print("=" * 65)
    
    repo_id = "Dev4285/MiniArt-2.0"
    print(f"[*] Target Repository: https://huggingface.co/{repo_id}")
    print("[*] Dataset Used: Qyrou/reasoning-corpus-4K-5M-v1")
    print("[*] Quantized GGUF Size: ~450 MB (< 1 GB Limit)")
    
    cmd = f"huggingface-cli upload {repo_id} . --repo-type=model"
    print(f"\n[>] Release Command:")
    print(f"    {cmd}")
    print("\n[+] Instructions:")
    print(" 1. Run 'huggingface-cli login' in terminal with your write token.")
    print(f" 2. Execute: {cmd}")
    print(f"\n[SUCCESS] Model package ready for Hugging Face upload!")

if __name__ == "__main__":
    main()
