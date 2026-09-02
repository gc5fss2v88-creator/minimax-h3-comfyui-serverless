import pathlib, re, sys

def main():
    import torch
    roots = [pathlib.Path(torch.__file__).parent / "lib"]
    found = set()
    for path in roots[0].glob("libtorch_cuda*.so*"):
        data = path.read_bytes()
        found |= {f"sm_{m.decode()}" for m in re.findall(rb"sm_(\d{2,3})[af]?", data)}
    required = sys.argv[1:] or ["sm_120"]
    print(f"torch={torch.__version__} cuda={torch.version.cuda} archs={sorted(found)}")
    missing = [x for x in required if x not in found]
    if missing:
        raise SystemExit(f"missing compiled CUDA architectures: {missing}")

if __name__ == "__main__":
    main()

