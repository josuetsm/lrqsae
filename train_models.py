from __future__ import annotations

from itertools import product

from mlx.utils import tree_flatten
from model import LowRankQuadraticSparseAutoencoder


# ----------------------------
# Shared hyperparameters
# ----------------------------
INPUT_DIM = 512
HIDDEN_DIM = 2048
LATENT_DIM = 512

K = 16
TAU = 17.5

FIT_KWARGS = dict(
    batch_size=1024,
    batches_per_block=100,
    n_epochs=16,
    total_len=9_932_800,
    zarr_path="data/Pythia70M-L3-res-wiki.zarr",
)


def count_trainable_params(model) -> int:
    """Return the total number of trainable scalar parameters."""
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def main() -> None:
    # We train 9 models: encoder_rank ∈ {0,1,2} × decoder_rank ∈ {0,1,2}
    rank_grid = [0, 1, 2]

    for encoder_rank, decoder_rank in product(rank_grid, rank_grid):
        sae = LowRankQuadraticSparseAutoencoder(
            INPUT_DIM,
            HIDDEN_DIM,
            LATENT_DIM,
            k=K,
            tau=TAU,
            encoder_rank=encoder_rank,
            decoder_rank=decoder_rank,
        )

        n_params = count_trainable_params(sae)
        print(
            f"[train] encoder_rank={encoder_rank} decoder_rank={decoder_rank} "
            f"trainable_params={n_params:,}"
        )

        sae.fit(**FIT_KWARGS)


if __name__ == "__main__":
    main()
