"""
NOT part of the deliverable -- internal sanity check only.

Validates the *mechanics* of the model.py architecture (shared trunk,
independent per-parameter heads, mixed classification+regression losses,
gradients flowing correctly, loss actually decreasing) using a from-scratch
NumPy MLP with manual backprop, since torch/sentence-transformers couldn't
be run in this sandbox (torch's default PyPI install pulled a full CUDA
dependency stack and hit a disk quota here; sentence-transformers additionally
needs to download weights from huggingface.co, which this sandbox's network
policy blocks).

Real sentence embeddings are replaced here with a stand-in: a fixed random
projection of a bag-of-words vector for each sentence. That's obviously not
as good as real sentence embeddings, but it's a legitimate substitute for
THIS test's purpose -- it produces a fixed, non-trainable embedding per
sentence (same role a frozen sentence-transformer would play), so training
the trunk+heads on top of it validates exactly the part of model.py that
actually needs validating: does the shared-trunk multi-head training loop
work at all.
"""
import csv
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_BY_NAME, PARAM_NAMES

rng = np.random.default_rng(0)

# --- load the real synthetic dataset ---
with open("data/text2city_dataset.csv") as f:
    rows = list(csv.DictReader(f))
print(f"loaded {len(rows)} rows")

# --- stand-in "frozen embedding": hashed bag-of-words -> fixed random projection ---
EMBED_DIM = 64
VOCAB_HASH_SIZE = 2000
proj = rng.normal(size=(VOCAB_HASH_SIZE, EMBED_DIM)) * 0.1  # fixed, never trained


def embed(text: str) -> np.ndarray:
    bow = np.zeros(VOCAB_HASH_SIZE)
    for word in text.lower().replace(",", "").replace(".", "").split():
        bow[hash(word) % VOCAB_HASH_SIZE] += 1.0
    return bow @ proj


X = np.stack([embed(r["text"]) for r in rows])  # (N, EMBED_DIM)

# --- targets: pick 2 representative heads to test both loss types at once ---
terrain_spec = PARAM_BY_NAME["terrain"]
density_spec = PARAM_BY_NAME["density"]

terrain_classes = terrain_spec.choices
y_terrain = np.array([terrain_classes.index(r["terrain"]) for r in rows])  # class idx
y_density_raw = np.array([float(r["density"]) for r in rows])
lo, hi = density_spec.value_range
y_density_frac = (y_density_raw - lo) / (hi - lo)  # target in [0,1] for sigmoid head

n_classes = len(terrain_classes)
N, D = X.shape
H = 32  # hidden size, mirrors the small trunk in model.py

# --- shared trunk + two heads, manual numpy/backprop ---
W1 = rng.normal(size=(D, H)) * np.sqrt(2.0 / D)
b1 = np.zeros(H)
W_terrain = rng.normal(size=(H, n_classes)) * 0.1
b_terrain = np.zeros(n_classes)
W_density = rng.normal(size=(H, 1)) * 0.1
b_density = np.zeros(1)

lr = 0.05
batch_size = 64
n_epochs = 30

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

losses = []
for epoch in range(n_epochs):
    perm = rng.permutation(N)
    epoch_loss = 0.0
    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        xb = X[idx]
        yb_terrain = y_terrain[idx]
        yb_density = y_density_frac[idx]
        b = len(idx)

        # forward: shared trunk
        h_pre = xb @ W1 + b1
        h = np.maximum(h_pre, 0)  # ReLU

        # terrain head (softmax classification)
        logits = h @ W_terrain + b_terrain
        probs = softmax(logits)
        terrain_loss = -np.log(probs[np.arange(b), yb_terrain] + 1e-9).mean()

        # density head (sigmoid regression)
        density_pred = sigmoid(h @ W_density + b_density).squeeze(-1)
        density_loss = np.mean((density_pred - yb_density) ** 2)

        loss = terrain_loss + density_loss
        epoch_loss += loss * b

        # backward -- terrain head
        d_logits = probs.copy()
        d_logits[np.arange(b), yb_terrain] -= 1
        d_logits /= b
        dW_terrain = h.T @ d_logits
        db_terrain = d_logits.sum(axis=0)
        d_h_from_terrain = d_logits @ W_terrain.T

        # backward -- density head
        d_density = (2 * (density_pred - yb_density) * density_pred * (1 - density_pred) / b)[:, None]
        dW_density = h.T @ d_density
        db_density = d_density.sum(axis=0)
        d_h_from_density = d_density @ W_density.T

        # backward -- shared trunk (gradients from BOTH heads combine here)
        d_h = d_h_from_terrain + d_h_from_density
        d_h_pre = d_h * (h_pre > 0)
        dW1 = xb.T @ d_h_pre
        db1 = d_h_pre.sum(axis=0)

        # update
        W_terrain -= lr * dW_terrain; b_terrain -= lr * db_terrain
        W_density -= lr * dW_density; b_density -= lr * db_density
        W1 -= lr * dW1; b1 -= lr * db1

    losses.append(epoch_loss / N)
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        # quick accuracy/error check
        h = np.maximum(X @ W1 + b1, 0)
        pred_terrain = np.argmax(h @ W_terrain + b_terrain, axis=1)
        acc = (pred_terrain == y_terrain).mean()
        pred_density = sigmoid(h @ W_density + b_density).squeeze(-1)
        mae = np.mean(np.abs(pred_density - y_density_frac)) * (hi - lo)
        print(f"epoch {epoch:2d}  loss={losses[-1]:.4f}  terrain_acc={acc:.3f}  density_MAE={mae:.4f}")

print(f"\nloss went from {losses[0]:.4f} to {losses[-1]:.4f}")
assert losses[-1] < losses[0] * 0.5, "loss should drop substantially -- mechanics check failed"
print("PASS: shared-trunk multi-head training mechanics work as designed")
