import os, subprocess, sys
if __name__ == "__main__":
    cmd = [
        sys.executable,
        "multilayer_perceptron.py",
        "-d",
        "newsgroups",
        "-e",
        "10",
        "-l",
        "0.001",
        "--remove_stopwords",
        "0",
        "--max_vocab_size",
        "50000",
        "--max_length",
        "300",
        "--batch_size",
        "8",
        "--ensemble_seeds",
        "0,1,2,3,4",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=os.path.dirname(__file__) or "."))
