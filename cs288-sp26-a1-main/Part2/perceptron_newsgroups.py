import os, subprocess, sys
if __name__ == "__main__":
    cmd = [sys.executable, "perceptron.py", "-d", "newsgroups", "-f", "bow+ngmeta+style", "-e", "3", "-l", "0.1"]
    raise SystemExit(subprocess.call(cmd, cwd=os.path.dirname(__file__) or "."))
