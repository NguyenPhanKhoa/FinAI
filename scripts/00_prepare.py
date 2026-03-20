"""
Step 0: Prepare environment for FinAI deployment.

Usage:
    python scripts/00_prepare.py           # Check everything
    python scripts/00_prepare.py --skip-docker  # Skip Docker check

This script:
1. Checks Python version (3.11+)
2. Checks Docker Desktop (optional for Docker method)
3. Creates .env file if missing
4. Checks HuggingFace token
5. Verifies hardware (NPU)

Supports both Docker and Native deployment methods.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_python():
    """Check Python version."""
    print("=" * 50)
    print("Checking Python...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 11:
        print("  [OK] Python 3.11+")
        return True
    else:
        print("  [FAIL] Python 3.11+ required. Download: https://python.org/downloads")
        return False


def check_docker():
    """Check Docker Desktop."""
    print("\n" + "=" * 50)
    print("Checking Docker...")

    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  [OK] Docker {version} installed")
            print("  Use Docker method: .\\run.ps1 -FullPipeline")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    print("  [WARN] Docker not found")
    print("  Option 1: Install Docker Desktop (recommended)")
    print("           https://docker.com/desktop")
    print("  Option 2: Use Native method (see Part 2B)")
    return False


def check_huggingface_token():
    """Check HF_TOKEN in .env file."""
    print("\n" + "=" * 50)
    print("Checking HuggingFace Token...")

    env_path = Path(__file__).parent.parent / ".env"

    # Check if .env exists
    if not env_path.exists():
        env_example = Path(__file__).parent.parent / ".env.example"
        if env_example.exists():
            print("  Creating .env from .env.example...")
            with open(env_example) as f:
                content = f.read()
            with open(env_path, "w") as f:
                f.write(content)
            print(f"  [CREATED] {env_path}")
        else:
            print(f"  [FAIL] .env not found and .env.example missing")
            return False

    # Read .env
    with open(env_path) as f:
        content = f.read()

    # Check for HF_TOKEN
    for line in content.split("\n"):
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if token and token != "your_token_here" and not token.startswith("#"):
                # Mask token
                masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
                print(f"  [OK] HF_TOKEN found: {masked}")
                return True
            else:
                print("  [WARN] HF_TOKEN is empty or placeholder")
                print(f"  Edit {env_path} and add your token")
                print("  Get token: https://huggingface.co/settings/tokens")
                return False

    print("  [WARN] HF_TOKEN not found in .env")
    print(f"  Edit {env_path} and add: HF_TOKEN=hf_...")
    print("  Get token: https://huggingface.co/settings/tokens")
    return False


def check_hardware():
    """Check hardware (NPU)."""
    print("\n" + "=" * 50)
    print("Checking Hardware...")

    # Try to run check_hardware.py
    script_path = Path(__file__).parent / "check_hardware.py"
    if script_path.exists():
        try:
            print("  Running scripts/check_hardware.py...")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True, timeout=30
            )
            # Show last 10 lines
            output = result.stdout
            if result.returncode == 0:
                print("  [OK] Hardware check passed")
                return True
            else:
                print("  [WARN] Hardware check has warnings")
                print("  (This is optional - FinAI can still run)")
                return True  # Not critical
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"  [SKIP] Could not run hardware check: {e}")
            return True
    else:
        print("  [SKIP] scripts/check_hardware.py not found")
        return True


def check_license():
    """Check if Llama license accepted."""
    print("\n" + "=" * 50)
    print("Checking Llama 3.1 License...")

    print("  You must accept license at:")
    print("  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("  Login → Accept License")
    print("  [ACTION REQUIRED] Open URL and accept if not done")
    return True  # Can't programmatically check this


def main():
    parser = argparse.ArgumentParser(description="Prepare FinAI environment")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker check")
    parser.add_argument("--skip-hardware", action="store_true", help="Skip hardware check")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  FinAI - Prepare Environment")
    print("=" * 50)

    results = {}

    # Check Python
    results["Python"] = check_python()

    # Check Docker (optional)
    if not args.skip_docker:
        results["Docker"] = check_docker()
    else:
        results["Docker"] = None

    # Check HuggingFace Token
    results["HF_TOKEN"] = check_huggingface_token()

    # Check License
    results["License"] = check_license()

    # Check Hardware (optional)
    if not args.skip_hardware:
        results["Hardware"] = check_hardware()

    # Summary
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)

    all_ok = True
    for name, status in results.items():
        if status is None:
            print(f"  {name}: [SKIP]")
        elif status:
            print(f"  {name}: [OK]")
        else:
            print(f"  {name}: [FIX NEEDED]")
            all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("  Ready to deploy!")
        print("  Next step: .\\run.ps1 -FullPipeline")
    else:
        print("  Please fix the issues above")
        print("  Then run this script again: python scripts/00_prepare.py")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
