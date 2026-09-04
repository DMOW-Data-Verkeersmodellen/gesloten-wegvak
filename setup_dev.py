#!/usr/bin/env python3
"""
Single entrypoint script to set up the local developer environment.
Installs package dependencies, git hooks, and nbwipers local filter.
"""

import os
import sys
import subprocess
from pathlib import Path
import tempfile

def run(cmd):
    print(f"--> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    print("Initializing developer environment...\n")

    # 1. Direct pre-commit cache to local C: drive on Windows to avoid UNC network lag
    if sys.platform == "win32":
        local_cache = Path(tempfile.gettempdir()) / "pre-commit-cache"
        local_cache.mkdir(parents=True, exist_ok=True)
        os.environ["PRE_COMMIT_HOME"] = str(local_cache)
        print(f"--> Configured PRE_COMMIT_HOME at {local_cache}\n")

    # 2. Install package in editable mode with dev dependencies
    run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

    # 3. Install pre-commit hooks using active Python interpreter
    run([sys.executable, "-m", "pre_commit", "install"])

    # 4. Pre-build hook environments into local cache (prevents commit delays later)
    print("\n--> Pre-building pre-commit environments on local drive...")
    run([sys.executable, "-m", "pre_commit", "install-hooks"])

    # 5. Register nbwipers as local git filter
    run(["nbwipers", "install", "local"])

    print("\n✅ Setup complete! Pre-commit hooks are pre-built on local C: drive.")

if __name__ == "__main__":
    main()