"""Securely install a Kaggle credential file. Usage: python setup_kaggle.py path/to/kaggle.json"""
from __future__ import annotations
import argparse, json, os, shutil, stat
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("credential", type=Path)
args = parser.parse_args()
data = json.loads(args.credential.read_text(encoding="utf-8"))
if not all(isinstance(data.get(k), str) and data[k].strip() for k in ("username", "key")):
    raise SystemExit("Credential must contain non-empty 'username' and 'key' strings.")
target_dir = Path.home() / ".kaggle"
target_dir.mkdir(parents=True, exist_ok=True)
target = target_dir / "kaggle.json"
shutil.copyfile(args.credential, target)
if os.name != "nt": target.chmod(stat.S_IRUSR | stat.S_IWUSR)
print(f"Kaggle credentials installed at {target}. Do not commit this file.")

