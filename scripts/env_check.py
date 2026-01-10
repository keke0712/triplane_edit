import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "face_parsing_pytorch"))


def which(cmd):
    return shutil.which(cmd) or ""

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[failed] {e}"

def main():
    repo_root = Path(__file__).resolve().parents[1]
    print("== Repo ==")
    print("repo_root:", repo_root)

    print("\n== Python ==")
    print("python:", sys.executable)
    print("version:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())

    print("\n== Tools on PATH ==")
    for c in ["git", "cmake", "ninja", "cl", "nvcc"]:
        p = which(c + ".exe" if os.name == "nt" else c)
        print(f"{c}: {p}")

    print("\n== CUDA (system) ==")
    print("CUDA_HOME:", os.environ.get("CUDA_HOME", ""))
    print("nvcc --version:\n", run("nvcc --version"))

    print("\n== PyTorch ==")
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch.cuda.is_available:", torch.cuda.is_available())
        print("torch.version.cuda:", torch.version.cuda)
        print("cudnn:", torch.backends.cudnn.version())
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print("device_count:", n)
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                print(f"gpu[{i}] name:", name)
                print(f"gpu[{i}] capability:", cap)
    except Exception as e:
        print("[Torch import failed]", e)

    print("\n== Quick import check ==")
    try:
        from configs.init_configs import get_parser
        from training.triplane_editing import TriplaneEditingPipeline
        parser = get_parser()
        opts = parser.parse_args(args=[])
        _ = TriplaneEditingPipeline(opts=opts, device=getattr(opts, "device", "cuda"), outdir=getattr(opts, "outdir", "outputs"))
        print("Import & pipeline init: OK")
    except Exception as e:
        print("Import & pipeline init: FAILED")
        print("Error:", repr(e))

if __name__ == "__main__":
    main()