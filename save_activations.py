from datasets import load_dataset
from tqdm.notebook import tqdm
from mlx import nn
import mlx.core as mx
from mlx_lm import load
from safetensors.numpy import save_file
import numpy as np
import zarr
from numcodecs import Blosc
import json
import os


# --------------------------------------------------------------------------------
# Utility: robustly retrieve embeddings and transformer layers
# --------------------------------------------------------------------------------
def get_embedding_and_layers(model):
    """
    Returns (embedding, layers) for a GPTNeoXModel in MLX-LM.

    Expected structure:
    - model.model: GPTNeoXModel
    - mm.embed_in: input embedding layer
    - mm.layers or mm.h: sequence of TransformerBlock objects
    """
    mm = model.model  # GPTNeoXModel

    # Input embedding
    if not hasattr(mm, "embed_in"):
        raise AttributeError("Expected attribute 'embed_in' in GPTNeoXModel")
    embed = mm.embed_in

    # Transformer layers
    if hasattr(mm, "layers"):
        layers = mm.layers
    elif hasattr(mm, "h"):
        layers = mm.h
    else:
        raise AttributeError("Could not find 'layers' nor 'h' in GPTNeoXModel")

    return embed, layers


def get_hidden_states(input_ids, layer_idx):
    """
    Compute hidden states up to a given transformer layer (inclusive).

    Parameters
    ----------
    input_ids : mx.array, shape [B, T], dtype int32
        Token IDs.
    layer_idx : int
        Zero-based layer index. The forward pass is applied up to and
        including this layer.
    """
    embed, layers = get_embedding_and_layers(model)

    n_layers = len(layers)
    assert 0 <= layer_idx < n_layers, (
        f"layer_idx={layer_idx} out of range (0..{n_layers-1})"
    )

    h = embed(input_ids)

    # Some implementations accept mask="causal", others do not
    for block in layers[: layer_idx + 1]:
        try:
            h = block(h, mask="causal")
        except TypeError:
            h = block(h)

    return h


def save_state(offset):
    """Persist current offset to disk for resumable execution."""
    with open(state_path, "w") as f:
        json.dump({"offset": offset}, f)


def load_state():
    """Load previously saved offset if available."""
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {"offset": 0}


# --------------------------------------------------------------------------------
# Model: Pythia-70M (deduped) loaded with MLX
# --------------------------------------------------------------------------------
model_name = "EleutherAI/pythia-70m-deduped"
model, tokenizer = load(model_name)

# Ensure a padding token exists
if tokenizer.pad_token is None:
    # GPT-NeoX typically uses EOS; reuse it as PAD
    tokenizer.pad_token = tokenizer.eos_token

pad_token_id = tokenizer.pad_token_id
bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id


# --------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------
dataset = load_dataset("wikipedia", "20220301.simple")

np.random.seed(42)
indices = np.random.permutation(len(dataset["train"]))


# --------------------------------------------------------------------------------
# Parameters (adapted to Pythia-70M)
# --------------------------------------------------------------------------------
layer = 3                # Pythia-70M has 6 layers (0..5). Layer 3 is mid-depth.
seq_len = 64
batch_size = 128
activation_dim = 512     # Hidden size of Pythia-70M

n_tokens_est = 10_000_000
zarr_path = "data/Pythia70M-L3-res-wiki.zarr"
input_ids_path = "data/Pythia70M-L3-res-wiki-token-ids.jsonl"
state_path = zarr_path + ".state.json"


# --------------------------------------------------------------------------------
# Open or create Zarr dataset
# --------------------------------------------------------------------------------
if not os.path.exists(zarr_path):
    print("Creating new Zarr dataset...")
    z = zarr.open(
        zarr_path,
        mode="w",
        shape=(n_tokens_est, activation_dim),
        dtype="f2",  # float16
        chunks=(10_000, activation_dim),
        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
        zarr_format=2,
    )
    state = {"offset": 0}
else:
    print("Resuming existing Zarr dataset...")
    z = zarr.open(zarr_path, mode="r+")
    state = load_state()


# --------------------------------------------------------------------------------
# Resume logic with optional dry run to align offsets
# --------------------------------------------------------------------------------
target_offset = state["offset"]

dry_run = False
offset = 0
buffer = []

try:
    with tqdm(indices, total=len(indices)) as pbar:
        for i, idx in enumerate(pbar):
            text = dataset["train"][idx.item()]["text"]

            # For Pythia, no need to manually remove BOS; tokenizer does not prepend it
            tokens = tokenizer.encode(text)

            for j in range(0, len(tokens), seq_len - 1):
                chunk = tokens[j : j + (seq_len - 1)]
                if len(chunk) < (seq_len - 1):
                    chunk = chunk + [pad_token_id] * ((seq_len - 1) - len(chunk))

                # Insert BOS token at the beginning
                chunk = [bos_token_id] + chunk
                buffer.append(chunk)

                if len(buffer) >= batch_size:
                    batch = buffer[:batch_size]
                    buffer = buffer[batch_size:]

                    if dry_run:
                        # Only count how many valid tokens would be processed
                        input_ids = np.array(batch, dtype=np.int32)
                        flat = input_ids.reshape(-1)
                        valid_mask = (flat != pad_token_id) & (flat != bos_token_id)
                        n_valid = int(valid_mask.sum())
                        offset += n_valid

                        if offset >= target_offset:
                            print(
                                f"✅ Dry run finished at offset={offset}, resuming actual storage"
                            )
                            dry_run = False
                            offset = target_offset  # Align exactly
                    else:
                        # Compute and store activations
                        input_ids = mx.array(batch, dtype=mx.int32)

                        x = get_hidden_states(input_ids, layer)

                        # Select only valid (non-BOS, non-PAD) tokens
                        input_ids_np = np.array(batch, dtype=np.int32).reshape(-1)
                        valid_mask = (
                            (input_ids_np != pad_token_id)
                            & (input_ids_np != bos_token_id)
                        )
                        valid_indices = np.where(valid_mask)[0]

                        x = x.reshape(-1, activation_dim)
                        x = x[valid_indices.tolist()]

                        arr = np.asarray(x, dtype=np.float16)
                        n = arr.shape[0]

                        z[offset : offset + n, :] = arr
                        offset += n

                        # Also store valid token IDs
                        with open(input_ids_path, "a") as f:
                            for row in batch:
                                valid_tids = [
                                    int(t)
                                    for t in row
                                    if t not in (pad_token_id, bos_token_id)
                                ]
                                f.write(json.dumps(valid_tids) + "\n")

            if i % 100 == 0 and not dry_run:
                save_state(offset)

            pbar.set_postfix({"n_tokens": offset})

except KeyboardInterrupt:
    print("\n⚠️ Interrupted manually. Saving state...")
    save_state(offset)
