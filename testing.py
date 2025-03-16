import subprocess, os

TEXT = "email"
DO_WANDB = True

if DO_WANDB: 
    try: 
        import wandb
    except ImportError: 
        DO_WANDB = False

def compress_rate(inp: str): 
    out_path = "tests/lm_output.bins"
    subprocess.run(["python3", "lm_to_compressor.py", "-c", inp, "tests/lm_output.bins"])
    in_size = os.path.getsize(inp)
    out_size = sum([os.path.getsize(os.path.join(out_path, f)) for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))])
    print(in_size, out_size)
    if DO_WANDB: wandb.log({"input_size": in_size, "output_size": out_size, "compression_rate": out_size/in_size})
    return out_size / in_size

if DO_WANDB: 
    run = wandb.init(
        entity="andy-and-only-andy",
        project="lm-data-compressor",
        config={
            "input": TEXT,
            "notes": ""
        },
    )
    

rate = compress_rate(f"texts/{TEXT}.txt")
print("Compression rate:": rate)