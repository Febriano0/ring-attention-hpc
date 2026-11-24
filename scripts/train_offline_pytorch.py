#!/usr/bin/env python3
"""
Pelatihan offline model Transformer minimal (1 layer single-head attention, d=64,
tanpa positional encoding absolut, mean pooling, dan classifier 2 kelas) pada CPU
menggunakan PyTorch, serta ekspor bobot FP32 row-major untuk inferensi C++/MPI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

VOCAB_SIZE = 4098


class BreastTriageTransformer(nn.Module):
    """Minimal single-head attention classifier matching Proposal Section 2.3."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab_size, d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, num_classes, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)  # [B, S, d]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d)
        probs = torch.softmax(scores, dim=-1)
        h = torch.matmul(probs, v)
        z = self.wo(h)
        pooled = z.mean(dim=1)
        return self.fc(pooled)


def load_windows(manifest: List[Dict[str, object]], seq_len: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load non-overlapping seq_len token windows per sample with train/val/lolo tags."""
    all_x: List[np.ndarray] = []
    all_y: List[int] = []
    all_samples: List[str] = []

    for entry in manifest:
        label = int(entry["label"])
        if label < 0:
            continue
        arr = np.fromfile(str(entry["tokens_path"]), dtype="<i4")
        n_win = len(arr) // seq_len
        for w in range(n_win):
            all_x.append(arr[w * seq_len : (w + 1) * seq_len])
            all_y.append(label)
            all_samples.append(str(entry["sample"]))

    return np.stack(all_x).astype(np.int64), np.asarray(all_y, dtype=np.int64), all_samples


def compute_binary_f1(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    acc = float((preds == targets).mean())
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    denom = 2 * tp + fp + fn
    f1 = (2.0 * tp / denom) if denom > 0 else 0.0
    return acc, f1


def export_weights_fp32(model: BreastTriageTransformer, out_dir: Path) -> None:
    """Export trained tensors in raw little-endian FP32 row-major binary format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "embed.bin": model.embed.weight.detach().cpu().numpy().astype("<f4"),
        "wq.bin": model.wq.weight.detach().cpu().numpy().astype("<f4"),
        "wk.bin": model.wk.weight.detach().cpu().numpy().astype("<f4"),
        "wv.bin": model.wv.weight.detach().cpu().numpy().astype("<f4"),
        "wo.bin": model.wo.weight.detach().cpu().numpy().astype("<f4"),
        "fc_w.bin": model.fc.weight.detach().cpu().numpy().astype("<f4"),
        "fc_b.bin": model.fc.bias.detach().cpu().numpy().astype("<f4"),
    }
    for fname, arr in tensors.items():
        np.ascontiguousarray(arr).tofile(out_dir / fname)


@torch.no_grad()
def eval_reference_prefix(
    model: BreastTriageTransformer,
    tokens_path: Path,
    seq_len: int,
) -> Dict[str, object]:
    arr = np.fromfile(tokens_path, dtype="<i4")[:seq_len].astype(np.int64)
    inp = torch.from_numpy(arr).unsqueeze(0)
    logits = model(inp).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    pred = int(torch.argmax(probs).item())
    return {
        "input": str(tokens_path),
        "s": int(arr.size),
        "predicted_class": pred,
        "logits": [float(logits[0].item()), float(logits[1].item())],
        "probs": [float(probs[0].item()), float(probs[1].item())],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train minimal breast DNA triage Transformer on CPU.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tokens"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/breast_triage_d64"))
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512, help="Short training context length <= 2048")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = args.data_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    x_all, y_all, samples = load_windows(manifest, args.seq_len)

    # Pemisahan per bagian sampel + leave-one-line-out untuk kelas progresi (184AA2_aig)
    train_idx: List[int] = []
    val_idx: List[int] = []
    lolo_idx: List[int] = []

    by_sample: Dict[str, List[int]] = {}
    for i, s_name in enumerate(samples):
        by_sample.setdefault(s_name, []).append(i)

    for s_name, idxs in by_sample.items():
        if s_name == "184AA2_aig":
            lolo_idx.extend(idxs)
        else:
            split = max(1, int(0.75 * len(idxs)))
            train_idx.extend(idxs[:split])
            val_idx.extend(idxs[split:])

    x_train = torch.from_numpy(x_all[train_idx])
    y_train = torch.from_numpy(y_all[train_idx])
    x_val = torch.from_numpy(x_all[val_idx])
    y_val = y_all[val_idx]
    x_lolo = torch.from_numpy(x_all[lolo_idx])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    model = BreastTriageTransformer(vocab_size=VOCAB_SIZE, d=args.d, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * bx.size(0)
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_preds = model(x_val).argmax(dim=-1).numpy()
                val_acc, val_f1 = compute_binary_f1(val_preds, y_val)
            print(f"[train] epoch={epoch:02d}/{args.epochs} loss={total_loss / len(train_idx):.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(x_val).argmax(dim=-1).numpy()
        val_acc, val_f1 = compute_binary_f1(val_preds, y_val)
        lolo_preds = model(x_lolo).argmax(dim=-1).numpy()
        lolo_acc = float((lolo_preds == y_all[lolo_idx]).mean()) if lolo_idx else 0.0

    export_weights_fp32(model, args.out_dir)

    refs = {}
    for name in ["184D_normal", "184AA2_aig"]:
        tpath = args.data_dir / f"{name}.bin"
        if tpath.exists():
            refs[name] = eval_reference_prefix(model, tpath, 1024)

    cfg = {
        "vocab_size": VOCAB_SIZE,
        "d": args.d,
        "num_classes": 2,
        "train_seq_len": args.seq_len,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "leave_one_line_out_184AA2_acc": lolo_acc,
        "class_names": {"0": "Normal-like", "1": "Progresi Karsinogenesis"},
        "limitations": (
            "Dataset PRJNA913438 hanya memiliki 1 lini normal (184D) dan 7 lini progresi; "
            "evaluasi akurasi merupakan metrik sekunder untuk demonstrasi aplikasi, bukan klaim klinis."
        ),
        "pytorch_reference_s1024": refs,
    }
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"[done] Bobot FP32 diekspor ke {args.out_dir} (val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, lolo_184AA2_acc={lolo_acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
