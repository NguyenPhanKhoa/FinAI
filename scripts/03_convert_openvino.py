"""
Step 3: Convert the merged model to OpenVINO IR format with INT4 symmetric quantization.

This uses optimum-cli under the hood to produce an NPU-compatible model.
Quantization settings:
  - INT4 symmetric (required for NPU)
  - ratio 1.0 (all weights in 4-bit)
  - group_size -1 (channel-wise, recommended for 7B+ models)
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)

    project_root = Path(__file__).parent.parent
    merged_path = project_root / config["paths"]["merged_model_dir"]
    output_path = project_root / config["paths"]["openvino_model_dir"]

    if not merged_path.exists():
        print("ERROR: Merged model not found. Run 02_merge_lora.py first.")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    quant = config["quantization"]
    cmd = [
        sys.executable, "-m", "optimum.exporters.openvino",
        "--model", str(merged_path),
        "--task", "text-generation-with-past",
        "--weight-format", quant["weight_format"],
        "--ratio", str(quant["ratio"]),
        "--group-size", str(quant["group_size"]),
        str(output_path),
    ]

    if quant.get("symmetric"):
        cmd.insert(-1, "--sym")

    print("Converting to OpenVINO IR with INT4 quantization...")
    print(f"  Source:  {merged_path}")
    print(f"  Output:  {output_path}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Conversion failed.")
        sys.exit(1)

    print("\nConversion complete.")
    print(f"OpenVINO model saved to: {output_path}")
    print("Next step: python scripts/04_run_inference.py")


if __name__ == "__main__":
    main()
