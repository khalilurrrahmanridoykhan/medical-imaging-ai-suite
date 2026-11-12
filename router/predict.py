"""CLI entry point: dispatch an input file to the right modality model.

Usage:
    python -m router.predict --input sample_xray.png --modality xray
"""

import argparse
import json
import sys

from router.registry import MODALITY_REGISTRY, available_modalities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a modality-specific prediction.")
    parser.add_argument("--input", required=True, help="Path to the input file.")
    parser.add_argument("--modality", required=True, help="Which modality model to run.")
    args = parser.parse_args()

    if args.modality not in MODALITY_REGISTRY:
        available = available_modalities() or ["none registered yet"]
        sys.exit(f"Unknown modality '{args.modality}'. Available: {available}")

    result = MODALITY_REGISTRY[args.modality](args.input)
    print(json.dumps({"modality": args.modality, **result}))


if __name__ == "__main__":
    main()
