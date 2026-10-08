import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_model import GPUTransformerModel, verify_aurelius_parameters
from aurelius_dataset import AureliusTokenDataset

TOKENS = "/content/drive2/MyDrive/Aurelius/data/corpus/aurelius_tokens.bin"
TRAIN_END = int((os.path.getsize(TOKENS) // 2) * 0.95)  # train=[0,TRAIN_END), val=[TRAIN_END,end)

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True)
p.add_argument("--steps", type=int, default=0)       # 0 = one full epoch
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--min-lr", type=float, default=None)  # None = constant after warmup
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--eval-every", type=int, default=5000)
p.add_argument("--save-every", type=int, default=5000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
p.add_argument("--no-save", action="store_true")
a = p.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)
train_ds = AureliusTokenDataset(TOKENS, 256, 0, TRAIN_END)
val_ds = AureliusTokenDataset(TOKENS, 256, TRAIN_END, None)
perm = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(a.seed))
epoch_steps = len(train_ds) // a.batch_size
total = a.steps or epoch_steps
min_lr = a.lr if a.min_lr is None else a.min_lr

def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    if min_lr >= a.lr:
        return a.lr
    pr = min(1.0, (s - a.warmup) / max(1, total - a.warmup))
    return min_lr + 0.5 * (a.lr - min_lr) * (1 + math.cos(math.pi * pr))

def get_batch(ds, idxs):
    items = [ds[int(i)] for i in idxs]
    return torch.stack([i[0] for i in items]), torch.stack([i[1] for i in items])

@torch.no_grad()
def evaluate(model, n_batches=100, bs=32):
    model.eval()
    idx = np.linspace(0, len(val_ds) - 1, n_batches * bs).astype(int)
    tot = 0.0
    for b in range(n_batches):
        x, y = get_batch(val_ds, idx[b * bs:(b + 1) * bs])
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x.to(dev))
        tot += F.cross_entropy(lg.view(-1, lg.size(-1)).float(), y.to(dev).view(-1)).item()
    model.train()
    return tot / n_batches

def save(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    assert os.path.getsize(path) > 1_000_000, f"bad checkpoint {path}"

os.makedirs(a.run_dir, exist_ok=True)
latest = os.path.join(a.run_dir, "aurelius_latest.pt")
bestp = os.path.join(a.run_dir, "aurelius_best.pt")
model = GPUTransformerModel().to(dev)
verify_aurelius_parameters(model)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
scaler = torch.amp.GradScaler()
step, best = 0, float("inf")
if a.resume:
    ck = torch.load(latest, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"]); step, best = ck["step"], ck["best_val"]
print(f"params ok | train ex {len(train_ds):,} | epoch_steps {epoch_steps:,} | "
      f"total {total:,} | start step {step} | warmup {a.warmup} | lr {a.lr}", flush=True)

logf = open(os.path.join(a.run_dir, "log.jsonl"), "a")
model.train(); run = []; t0 = time.time()
while step < total:
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    ids = perm[(step * a.batch_size + torch.arange(a.batch_size)) % len(perm)]
    x, y = get_batch(train_ds, ids)
    x, y = x.to(dev), y.to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        lg = model(x)
    loss = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), y.view(-1))
    if not math.isfinite(loss.item()):
        raise RuntimeError(f"non-finite loss at step {step}")
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    step += 1
    run.append(loss.item())

    if step % 100 == 0:
        rec = {"step": step, "train_loss": round(sum(run) / len(run), 4),
               "lr": round(lr_at(step - 1), 7), "min": round((time.time() - t0) / 60, 1)}
        run = []
        print(rec, flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()

    if step % a.eval_every == 0 or step == total:
        v = evaluate(model)
        improved = v < best
        best = min(best, v)
        print(f"VAL step {step}: {v:.4f} | best {best:.4f}", flush=True)
        logf.write(json.dumps({"step": step, "val_loss": round(v, 4)}) + "\n"); logf.flush()
        if improved and not a.no_save:
            save(bestp, {"model": model.state_dict(), "step": step, "val_loss": v})

    if not a.no_save and (step % a.save_every == 0 or step == total):
        save(latest, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                      "scaler": scaler.state_dict(), "step": step,
                      "best_val": best, "config": vars(a)})
        print(f"saved latest at step {step}", flush=True)
print("DONE", flush=True)
