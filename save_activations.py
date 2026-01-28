from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import zarr
from datasets import load_dataset
from mlx_lm import load
import mlx.core as mx
from numcodecs import Blosc
from tqdm.notebook import tqdm


# =============================================================================
# Utilities: model access (robust) + forward to a given layer
# =============================================================================
def get_embedding_and_layers(model) -> Tuple[object, Sequence[object]]:
    """
    Returns (embedding, layers) for a GPTNeoXModel in MLX-LM.

    Expected structure:
    - model.model: GPTNeoXModel
    - mm.embed_in: input embedding layer
    - mm.layers or mm.h: sequence of TransformerBlock objects
    """
    mm = model.model

    if not hasattr(mm, "embed_in"):
        raise AttributeError("Expected attribute 'embed_in' in GPTNeoXModel")
    embed = mm.embed_in

    if hasattr(mm, "layers"):
        layers = mm.layers
    elif hasattr(mm, "h"):
        layers = mm.h
    else:
        raise AttributeError("Could not find 'layers' nor 'h' in GPTNeoXModel")

    return embed, layers


def get_hidden_states(model, input_ids: mx.array, layer_idx: int) -> mx.array:
    """
    Compute hidden states up to a given transformer layer (inclusive).

    input_ids: mx.array [B, T] int32
    layer_idx: 0-based, applied up to and including layer_idx
    """
    embed, layers = get_embedding_and_layers(model)
    n_layers = len(layers)
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx={layer_idx} out of range (0..{n_layers - 1})")

    h = embed(input_ids)
    for block in layers[: layer_idx + 1]:
        try:
            h = block(h, mask="causal")
        except TypeError:
            h = block(h)
    return h


# =============================================================================
# State / metadata for resumable execution
# =============================================================================
@dataclass(frozen=True)
class RunMeta:
    model_name: str
    dataset_name: str
    dataset_config: str
    layer: int
    seq_len: int
    batch_size: int
    activation_dim: int
    pad_token_id: int
    bos_token_id: int
    zarr_chunks_tokens: int
    zarr_format: int
    dtype: str


def save_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_state(state_path: str, offset: int) -> None:
    save_json(state_path, {"offset": int(offset)})


def load_state(state_path: str) -> dict:
    return load_json(state_path, {"offset": 0})


# =============================================================================
# Main config
# =============================================================================
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DATASET_NAME = "wikipedia"
DATASET_CONFIG = "20220301.simple"

LAYER = 3
SEQ_LEN = 64
BATCH_SIZE = 128
ACTIVATION_DIM = 512  # Pythia-70M hidden size

N_TOKENS_EST = 11_000_000
ZARR_PATH = "data/Pythia70M-L3-res-wiki.zarr"
INPUT_IDS_PATH = "data/Pythia70M-L3-res-wiki-token-ids.jsonl"
STATE_PATH = ZARR_PATH + ".state.json"
META_PATH = ZARR_PATH + ".meta.json"

ZARR_CHUNKS_TOKENS = 10_000
ZARR_FORMAT = 2
DTYPE = "f2"


# =============================================================================
# Load model + tokenizer
# =============================================================================
model, tokenizer = load(MODEL_NAME)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pad_token_id = tokenizer.pad_token_id
bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

# (Recommended) ensure deterministic inference behavior for activations
if hasattr(model, "eval"):
    model.eval()


# =============================================================================
# Load dataset + shuffle indices
# =============================================================================
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
np.random.seed(42)
indices = np.random.permutation(len(dataset["train"]))


# =============================================================================
# Create/open Zarr + validate meta
# =============================================================================
run_meta = RunMeta(
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    dataset_config=DATASET_CONFIG,
    layer=LAYER,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    activation_dim=ACTIVATION_DIM,
    pad_token_id=int(pad_token_id),
    bos_token_id=int(bos_token_id),
    zarr_chunks_tokens=ZARR_CHUNKS_TOKENS,
    zarr_format=ZARR_FORMAT,
    dtype=DTYPE,
)

if not os.path.exists(ZARR_PATH):
    print("Creating new Zarr dataset...")
    z = zarr.open(
        ZARR_PATH,
        mode="w",
        shape=(N_TOKENS_EST, ACTIVATION_DIM),
        dtype=DTYPE,
        chunks=(ZARR_CHUNKS_TOKENS, ACTIVATION_DIM),
        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
        zarr_format=ZARR_FORMAT,
    )
    save_state(STATE_PATH, 0)
    save_json(META_PATH, asdict(run_meta))
else:
    print("Resuming existing Zarr dataset...")
    z = zarr.open(ZARR_PATH, mode="r+")
    state = load_state(STATE_PATH)
    meta_prev = load_json(META_PATH, None)

    # Basic meta guardrails (avoid silently mixing incompatible runs)
    if meta_prev is not None:
        for k in ["model_name", "layer", "seq_len", "activation_dim", "pad_token_id", "bos_token_id", "dtype"]:
            if meta_prev.get(k) != asdict(run_meta).get(k):
                raise ValueError(
                    f"Incompatible resume: meta[{k}]={meta_prev.get(k)} != current[{k}]={asdict(run_meta).get(k)}"
                )

    if int(state.get("offset", 0)) < 0:
        raise ValueError("State offset is negative; state file is corrupted?")

# Load offset (after create/resume)
state = load_state(STATE_PATH)
offset = int(state["offset"])


# =============================================================================
# Optional: dry-run alignment (only if you suspect state drift)
# =============================================================================
DRY_RUN_ALIGN = False  # set True if you want to recompute offset by scanning tokens
target_offset = offset


# =============================================================================
# Extraction loop
# =============================================================================
buffer = []
t0 = time.time()
last_save = time.time()
last_offset = offset

try:
    with tqdm(indices, total=len(indices)) as pbar:
        for i, idx in enumerate(pbar):
            text = dataset["train"][int(idx)]["text"]
            tokens = tokenizer.encode(text)

            # Chunk into (SEQ_LEN - 1) and prepend BOS
            for j in range(0, len(tokens), SEQ_LEN - 1):
                chunk = tokens[j : j + (SEQ_LEN - 1)]
                if len(chunk) < (SEQ_LEN - 1):
                    chunk = chunk + [pad_token_id] * ((SEQ_LEN - 1) - len(chunk))
                chunk = [bos_token_id] + chunk
                buffer.append(chunk)

                if len(buffer) < BATCH_SIZE:
                    continue

                batch = buffer[:BATCH_SIZE]
                buffer = buffer[BATCH_SIZE:]

                input_np = np.asarray(batch, dtype=np.int32)          # [B, T]
                flat = input_np.reshape(-1)                           # [B*T]
                valid_mask = (flat != pad_token_id) & (flat != bos_token_id)
                n_valid = int(valid_mask.sum())

                if DRY_RUN_ALIGN:
                    offset += n_valid
                    if offset >= target_offset:
                        print(f"✅ Dry-run finished at offset={offset}; switching to actual storage.")
                        DRY_RUN_ALIGN = False
                        offset = target_offset
                    continue

                if n_valid == 0:
                    # Nothing to write for this batch
                    continue

                input_ids = mx.array(input_np, dtype=mx.int32)        # [B, T]

                x = get_hidden_states(model, input_ids, LAYER)        # [B, T, D]
                x = x.reshape(-1, ACTIVATION_DIM)                     # [B*T, D]

                valid_indices = np.where(valid_mask)[0].tolist()
                x = x[valid_indices]                                  # [n_valid, D]

                # Materialize before numpy conversion (prevents lazy eval surprises)
                mx.eval(x)

                arr = np.asarray(x, dtype=np.float16)
                n = int(arr.shape[0])

                z[offset : offset + n, :] = arr
                offset += n

                # Store token ids for the same valid positions
                # (Keep 1 jsonl line per *sequence* to match your prior format)
                with open(INPUT_IDS_PATH, "a") as f:
                    for row in batch:
                        valid_tids = [int(t) for t in row if t not in (pad_token_id, bos_token_id)]
                        f.write(json.dumps(valid_tids) + "\n")

                # Periodic state saving + basic throughput stats
                now = time.time()
                if now - last_save >= 30.0:
                    save_state(STATE_PATH, offset)
                    last_save = now

                dt = max(1e-6, now - t0)
                tps = (offset - last_offset) / max(1e-6, now - (now - (now - t0)))  # not super meaningful; keep simple below
                pbar.set_postfix({"tokens": offset, "elapsed_s": int(dt)})

            # Less frequent save by document index as well
            if (i % 100) == 0 and not DRY_RUN_ALIGN:
                save_state(STATE_PATH, offset)

except KeyboardInterrupt:
    print("\n⚠️ Interrupted manually. Saving state...")
    save_state(STATE_PATH, offset)
finally:
    # Always persist the last known offset
    if not DRY_RUN_ALIGN:
        save_state(STATE_PATH, offset)

print(f"Done. Final offset={offset}")
