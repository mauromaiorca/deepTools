#!/usr/bin/env python

import json
import os

CONFIG_PATH = os.path.expanduser("~/.deeptools_config.json")


def load_user_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def main():
    print("deepTools setup")
    print("=" * 40)

    existing = load_user_config()
    current = existing.get("models_dir", "")

    if current:
        print(f"Current models directory: {current}")
        print()

    models_dir = input("Enter the path to the learning_models directory\n"
                       "(leave empty to keep current): ").strip()

    if not models_dir and current:
        models_dir = current

    if not models_dir:
        print("No path provided. Nothing saved.")
        return

    models_dir = os.path.abspath(os.path.expanduser(models_dir))

    if not os.path.isdir(models_dir):
        print(f"Warning: directory does not exist: {models_dir}")
        confirm = input("Save anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    existing["models_dir"] = models_dir

    with open(CONFIG_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved to {CONFIG_PATH}")

    pth_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")] if os.path.isdir(models_dir) else []
    if pth_files:
        print(f"Found {len(pth_files)} model files in {models_dir}")
    else:
        print(f"No .pth files found in {models_dir} (you can add them later).")


if __name__ == "__main__":
    main()
