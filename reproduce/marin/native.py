"""Export recorded native-JAX settings for the two published Muon references.

This exports an inspection record, not a launch configuration: the original
Muon-enabled Levanter source is not identified by the recorded package version.
"""

import argparse
import json
from pathlib import Path


def export_settings(config: dict) -> dict:
    if config.get("engine") != "native_jax":
        raise ValueError("Choose one of the two native_jax reference configs.")
    return {
        "status": "recorded settings; original Muon-enabled runtime required",
        "reference_validation_loss": config["reference_validation_loss"],
        **config["native"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings = export_settings(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Exported recorded settings to {args.output}; this is not a launch config.")


if __name__ == "__main__":
    main()
