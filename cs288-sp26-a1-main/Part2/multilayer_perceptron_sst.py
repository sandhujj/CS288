import os, subprocess, sys

if __name__ == "__main__":

    cmd = [sys.executable, "multilayer_perceptron.py", "-d", "sst2", "-e", "3", "-l", "0.001"]

    raise SystemExit(subprocess.call(cmd, cwd=os.path.dirname(__file__) or "."))

