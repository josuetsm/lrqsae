import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import zarr
import os
import time
import pickle
import datetime
from tqdm.notebook import tqdm
from functools import partial
from mlx.utils import tree_flatten

# ==============================
#  BLOQUES BILINEALES
# ==============================

class BilinearEncoder(nn.Module):
    """
    Encoder bilineal de bajo rango:
        u(x) = W x + sum_{k=1}^rank (U_k x) ⊙ (V_k x)
    """
    def __init__(self, in_dim: int, latent_dim: int, rank: int = 0, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.rank = int(rank)

        self.linear = nn.Linear(in_dim, latent_dim, bias=bias)

        if self.rank > 0:
            scale = 1.0 / np.sqrt(in_dim)
            # [rank, latent_dim, in_dim]
            self.U = mx.random.normal(
                shape=(self.rank, latent_dim, in_dim),
                dtype=mx.float32
            ) * scale
            self.V = mx.random.normal(
                shape=(self.rank, latent_dim, in_dim),
                dtype=mx.float32
            ) * scale
        else:
            self.U = None
            self.V = None

    def _forward_2d(self, x):
        """
        x: [B, in_dim]
        devuelve: [B, latent_dim]
        """
        B, D = x.shape
        assert D == self.in_dim, f"Expected in_dim={self.in_dim}, got {D}"

        out = self.linear(x)  # [B, latent_dim]
        if self.rank == 0:
            return out

        # U, V: [rank, latent_dim, in_dim] -> [rank*latent_dim, in_dim]
        U_flat = mx.reshape(self.U, (self.rank * self.latent_dim, self.in_dim))
        V_flat = mx.reshape(self.V, (self.rank * self.latent_dim, self.in_dim))

        # x [B, in_dim] @ U_flat.T [in_dim, rank*latent_dim] -> [B, rank*latent_dim]
        u = mx.matmul(x, mx.transpose(U_flat))
        v = mx.matmul(x, mx.transpose(V_flat))

        # [B, rank*latent_dim] -> [B, rank, latent_dim]
        u = mx.reshape(u, (B, self.rank, self.latent_dim))
        v = mx.reshape(v, (B, self.rank, self.latent_dim))

        # sum_k (U_k x) ⊙ (V_k x): sum over axis=1 (rank)
        bil = (u * v).sum(axis=1)  # [B, latent_dim]

        return out + bil

    def __call__(self, x):
        """
        Soporta:
          - x: [B, in_dim]      -> [B, latent_dim]
          - x: [B, T, in_dim]   -> [B, T, latent_dim]
        """
        if x.ndim == 2:
            return self._forward_2d(x)

        if x.ndim == 3:
            B, T, D = x.shape
            assert D == self.in_dim, f"Expected in_dim={self.in_dim}, got {D}"

            x_flat = mx.reshape(x, (B * T, D))          # [B*T, in_dim]
            z_flat = self._forward_2d(x_flat)           # [B*T, latent_dim]
            z = mx.reshape(z_flat, (B, T, self.latent_dim))  # [B, T, latent_dim]
            return z

        raise ValueError(f"BilinearEncoder expects x.ndim in {{2,3}}, got {x.ndim}")



class BilinearDecoder(nn.Module):
    """
    Decoder lineal + warp bilineal en el espacio de salida:
        x*   = D z
        x̂ = x* + sum_{k=1}^rank (U_k x*) ⊙ (V_k x*)
    """
    def __init__(self, latent_dim: int, out_dim: int, rank: int = 0, bias: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.rank = int(rank)

        self.linear = nn.Linear(latent_dim, out_dim, bias=bias)

        if self.rank > 0:
            scale = 1.0 / np.sqrt(out_dim)
            self.U_d = mx.random.normal(shape=(self.rank, out_dim, out_dim), dtype=mx.float32) * scale
            self.V_d = mx.random.normal(shape=(self.rank, out_dim, out_dim), dtype=mx.float32) * scale
        else:
            self.U_d = None
            self.V_d = None

    def __call__(self, z):
        """
        z: [B, latent_dim]
        devuelve: x̂ [B, out_dim]
        """
        x_lin = self.linear(z)  # [B, out_dim]
        if self.rank == 0:
            return x_lin

        B = x_lin.shape[0]

        # U_d, V_d: [rank, out_dim, out_dim] -> [rank*out_dim, out_dim]
        U_flat = mx.reshape(self.U_d, (self.rank * self.out_dim, self.out_dim))
        V_flat = mx.reshape(self.V_d, (self.rank * self.out_dim, self.out_dim))

        # x_lin [B, out_dim] @ U_flat.T [out_dim, rank*out_dim] -> [B, rank*out_dim]
        u = mx.matmul(x_lin, mx.transpose(U_flat))
        v = mx.matmul(x_lin, mx.transpose(V_flat))

        # [B, rank*out_dim] -> [B, rank, out_dim]
        u = mx.reshape(u, (B, self.rank, self.out_dim))
        v = mx.reshape(v, (B, self.rank, self.out_dim))

        # sum_k (U_k x*) ⊙ (V_k x*): sum over axis=1 (rank)
        bil = (u * v).sum(axis=1)  # [B, out_dim]

        return x_lin + bil



# ==============================
#  AUTOENCODER ESPARSO BILINEAL REFINADO
# ==============================

class SparseAutoencoder(nn.Module):
    """
    SAE con:
      - Encoder lineal + bilineal de rango encoder_rank (en x).
      - Decoder lineal + bilineal de rango decoder_rank (en x*).
      - Sparsidad inducida por BatchTopK sobre z_pre.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int | None = None,
        lr: float = 1e-4,
        k: int = 64,
        lambda_mu: float = 0.1,
        tau: float = 8.0,
        seed: int = 42,
        encoder_rank: int = 0,
        decoder_rank: int = 0,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)

        self.lr = lr
        self.k = k
        self.lambda_mu = lambda_mu
        self.tau = tau
        self.seed = seed
        self.encoder_rank = int(encoder_rank)
        self.decoder_rank = int(decoder_rank)

        self.epochs_trained = 0
        self.batches_per_block = None

        mx.random.seed(seed)

        self.encoder = BilinearEncoder(
            in_dim=self.input_dim,
            latent_dim=self.latent_dim,
            rank=self.encoder_rank,
            bias=True,
        )
        self.decoder = BilinearDecoder(
            latent_dim=self.latent_dim,
            out_dim=self.output_dim,
            rank=self.decoder_rank,
            bias=True,
        )

        self.last_activation = mx.zeros((self.latent_dim,), dtype=mx.float32)
        self.optimizer = optim.AdamW(learning_rate=lr)

        self.history = {"mse": [], "dead_latents": []}

    def _arch_string(self):
        return f"{self.input_dim}_{self.latent_dim}_{self.output_dim}__Er{self.encoder_rank}_Dr{self.decoder_rank}"

    def save(self, path: str | None = None, save_datetime: bool = False):
        arch = self._arch_string()
        if save_datetime:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            base_name = f"bsae_{arch}_k{self.k}_ep{self.epochs_trained}_{ts}"
        else:
            base_name = f"bsae_{arch}_k{self.k}_ep{self.epochs_trained}"

        if path is None:
            os.makedirs("checkpoints", exist_ok=True)
            path = os.path.join("checkpoints", base_name)
        else:
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, base_name)

        checkpoint = {
            "weights": self.state.parameters(),
            "optimizer_state": self.optimizer.state,
            "history": self.history,
            "epochs_trained": self.epochs_trained,
            "config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "output_dim": self.output_dim,
                "k": self.k,
                "lambda_mu": self.lambda_mu,
                "tau": self.tau,
                "lr": self.lr,
                "seed": self.seed,
                "encoder_rank": self.encoder_rank,
                "decoder_rank": self.decoder_rank,
            },
        }
        with open(path + ".pkl", "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Checkpoint guardado en: {path}.pkl")

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "rb") as f:
            checkpoint = pickle.load(f)
        model = cls(**checkpoint["config"])
        model.load_weights(tree_flatten(checkpoint["weights"]))
        model.optimizer.state = checkpoint["optimizer_state"]
        model.history = checkpoint["history"]
        model.epochs_trained = checkpoint["epochs_trained"]
        return model

    # ==============================
    #  UTILIDADES
    # ==============================
    def clip_norms(self, x):
        norms = mx.linalg.norm(x, axis=1, keepdims=True)
        factors = mx.minimum(1.0, self.tau / norms)
        return x * factors

    def batch_topk(self, z, stride: int = 1):
        batch_k = self.k * z.shape[0] // stride
        thr = mx.topk(z[::stride].flatten(), batch_k).min()
        return mx.where(z >= thr, z, 0.0)

    def encode_pre(self, x):
        x = self.clip_norms(x)
        return self.encoder(x)

    def encode(self, x):
        z_pre = self.encode_pre(x)
        return self.batch_topk(z_pre)

    def decode(self, z):
        return self.decoder(z)

    def stats_reg(self, z_pre):
        mu = z_pre.mean(axis=0)
        sigma = z_pre.std(axis=0) + 1e-6
        mu_mean = mu.mean()
        sigma_mean = sigma.mean()
        loss_mu = ((mu - mu_mean) ** 2).mean()
        loss_sigma = ((sigma - sigma_mean) ** 2).mean()
        return loss_mu + loss_sigma

    def mse(self, x):
        x = self.clip_norms(x)
        z_pre = self.encoder(x)
        z = self.batch_topk(z_pre)
        x_hat = self.decoder(z)
        return mx.mean(((x - x_hat)**2) / x.var(axis=1, keepdims=True)).item()

    def loss_fn(self, x):
        x = self.clip_norms(x)
        z_pre = self.encoder(x)
        z = self.batch_topk(z_pre)
        x_hat = self.decoder(z)
        mse = mx.mean(((x - x_hat)**2) / x.var(axis=1, keepdims=True))

        activation_mask = z > 0
        self.last_activation += 1.0
        self.last_activation *= mx.where(activation_mask.any(0), 0.0, 1.0)

        return mse + self.lambda_mu * self.stats_reg(z_pre)

    def compile(self):
        self.loss_and_grad = nn.value_and_grad(self, self.loss_fn)
        tracked_state = [self.state, self.optimizer.state]
        self._train_step = partial(
            mx.compile,
            inputs=tracked_state,
            outputs=tracked_state
        )(self._train_step_impl)

    def _train_step_impl(self, x_batch):
        loss, grads = self.loss_and_grad(x_batch)
        self.optimizer.update(self, grads)
        return loss

    def train_step(self, x_batch):
        return self._train_step(x_batch)

    @staticmethod
    def batch_iterator(zarray, total_len, batch_size, batches_per_block):
        block_size = batch_size * batches_per_block
        for start in range(0, total_len, block_size):
            block = mx.array(zarray[start : start + block_size], dtype=mx.float32)
            perm = mx.random.permutation(mx.arange(len(block)))
            block = block[perm]
            for offset in range(0, len(block), batch_size):
                yield block[offset : offset + batch_size]

    def fit(
        self,
        batch_size=1024,
        batches_per_block=100,
        n_epochs=1,
        zarr_path='/path/to/your/data.zarr',
        total_len: int | None = None,
    ):
        zarray = zarr.open(zarr_path, mode='r')
        if total_len is None:
            total_len = len(zarray)
        num_blocks = total_len // (batch_size * batches_per_block)
        self.batches_per_block = batches_per_block
        self.compile()

        for epoch in tqdm(range(n_epochs), desc='Epochs', leave=True):
            total_loss, t0 = 0.0, time.time()
            with tqdm(total=num_blocks, desc=f"Epoch {epoch+1}", leave=False) as pbar:
                for i, x_batch in enumerate(self.batch_iterator(zarray, total_len, batch_size, batches_per_block)):
                    loss = self.train_step(x_batch)
                    total_loss += loss

                    if (i + 1) % batches_per_block == 0:
                        mse = self.mse(x_batch)
                        dead = (self.last_activation > batches_per_block).mean().item()
                        avg = (total_loss / batches_per_block).item()
                        print(
                            f"Epoch {epoch+1} | Block {(i+1)//batches_per_block}: "
                            f"loss={avg:.6f} | mse={mse:.6f} | dead={dead:.3f} | "
                            f"time={time.time()-t0:.2f}s"
                        )
                        self.history["mse"].append(mse)
                        self.history["dead_latents"].append(dead)
                        total_loss, t0 = 0.0, time.time()
                        pbar.update(1)

                    if (i + 1) % (batches_per_block * 10) == 0:
                        self.save()

            self.epochs_trained += 1
            self.save(save_datetime=True)