import os
import sys

def main():
    print("=" * 65)
    print("GITHUB REPOSITORY RELEASE - Dev4285/MiniArt-2.0")
    print("=" * 65)
    
    gh_repo = "Dev4285/MiniArt-2.0"
    print(f"[*] Target GitHub Repo: https://github.com/{gh_repo}")
    
    cmds = [
        "git init",
        "git add .",
        'git commit -m "Release MiniArt 2.0 (Reasoning + Vision SLM < 1GB)"',
        "git branch -M main",
        f"git remote add origin https://github.com/{gh_repo}.git",
        "git push -u origin main"
    ]
    
    print("\n[>] GitHub Release Commands:")
    for c in cmds:
        print(f"    {c}")
        
    print("\n[SUCCESS] Project structure ready for GitHub release!")

if __name__ == "__main__":
    main()
