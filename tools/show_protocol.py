"""Print the exact public parameter manifests and software versions."""
from pathlib import Path
import importlib.metadata
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print("Python:", platform.python_version(), "Platform:", sys.platform)
    for name in ("simulation/config.json", "config/empirical_protocol.json"):
        print("\nPARAMETERS:", name)
        print(json.dumps(json.loads((ROOT / name).read_text(encoding="utf-8")), indent=2))
    print("\nInstalled package versions:")
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        print(f"{name}=={actual}" + (" [matches]" if actual == expected else f" [expected {expected}]"))
