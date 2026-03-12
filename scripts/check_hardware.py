"""
Utility: Check hardware compatibility and OpenVINO device availability.
Run this first to verify your system is ready.
"""

import subprocess
import sys


def check_openvino():
    try:
        import openvino as ov
        print(f"OpenVINO version: {ov.__version__}")

        core = ov.Core()
        devices = core.available_devices
        print(f"Available devices: {devices}")

        for device in devices:
            full_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  {device}: {full_name}")

        if "NPU" in devices:
            print("\n[OK] NPU is available for inference.")
        else:
            print("\n[WARNING] NPU not detected by OpenVINO.")
            print("  Make sure Intel NPU driver is installed:")
            print("  https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html")

        return "NPU" in devices
    except ImportError:
        print("[ERROR] OpenVINO not installed. Run: pip install -r requirements.txt")
        return False


def check_openvino_genai():
    try:
        import openvino_genai
        print(f"OpenVINO GenAI version: {openvino_genai.__version__}")
        return True
    except ImportError:
        print("[ERROR] openvino-genai not installed. Run: pip install openvino-genai")
        return False


def check_system_info():
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    # CPU
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True
        )
        print(f"CPU: {result.stdout.strip()}")
    except Exception:
        print("CPU: (could not detect)")

    # RAM
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)"],
            capture_output=True, text=True
        )
        print(f"RAM: {result.stdout.strip()} GB")
    except Exception:
        print("RAM: (could not detect)")

    # NPU device
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             '(Get-PnpDevice | Where-Object { $_.FriendlyName -match "AI Boost|NPU" }).FriendlyName'],
            capture_output=True, text=True
        )
        npu = result.stdout.strip()
        print(f"NPU: {npu if npu else 'Not found'}")
    except Exception:
        print("NPU: (could not detect)")

    print()


def main():
    check_system_info()

    print("=" * 50)
    print("OPENVINO COMPATIBILITY CHECK")
    print("=" * 50)

    ov_ok = check_openvino()
    print()
    genai_ok = check_openvino_genai()

    print()
    print("=" * 50)
    if ov_ok and genai_ok:
        print("RESULT: System is ready for FinGPT NPU inference.")
    else:
        print("RESULT: Some dependencies are missing. See errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
