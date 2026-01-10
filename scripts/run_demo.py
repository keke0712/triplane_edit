import os, sys, argparse

def add_paths(repo_root: str):
    # 让 "import configs / training / ..." 能找到
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # 让 face_parsing_pytorch 里 "from resnet import ..." 这种写法能找到
    fp = os.path.join(repo_root, "face_parsing_pytorch")
    if os.path.isdir(fp) and fp not in sys.path:
        sys.path.insert(0, fp)

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    add_paths(repo_root)

    from configs.init_configs import get_parser
    from training.triplane_editing import TriplaneEditingPipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_base_dir", default="example")
    ap.add_argument("--src", required=True, help="e.g. 79.png")
    ap.add_argument("--dst", required=True, help="e.g. 40.png")
    ap.add_argument("--label", required=True, help="eyes/hair/mouth/nose ...")
    ap.add_argument("--runtime_optim", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    parser = get_parser()
    opts = parser.parse_args(args=[])

    # 覆盖常用参数
    opts.device = args.device
    opts.outdir = args.outdir

    pipe = TriplaneEditingPipeline(opts=opts, device=opts.device, outdir=opts.outdir)
    pipe.edit_demo(
        input_base_dir=args.input_base_dir,
        src_name=args.src,
        dst_name=args.dst,
        edit_label=args.label,
        runtime_optim=args.runtime_optim,
    )

if __name__ == "__main__":
    main()
