import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal, Tuple


# ────────────────────────────────────────────────────────────────────────────────
# Positional Encoding (from bandwidth-estimation)
# ────────────────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """Classic sine / cosine positional encoding (batch-first)."""

    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ────────────────────────────────────────────────────────────────────────────────
# Attention Pooling (from bandwidth-estimation)
# ────────────────────────────────────────────────────────────────────────────────
class MaskedAttentionPooling(nn.Module):
    """Self-attention pooling → single vector per sequence."""

    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            self.score_proj = nn.Linear(d_model, 1, bias=False)
        else:
            self.score_proj = nn.Sequential(
                nn.Linear(d_model, hidden, bias=True),
                nn.Tanh(),
                nn.Linear(hidden, 1, bias=False),
            )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feats : (B, T, D)
        mask  : (B, T) — True = PAD (to ignore)

        Returns
        -------
        pooled : (B, D)
        """
        scores = self.score_proj(feats).squeeze(-1)
        scores.masked_fill_(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bt, btd -> bd", attn, feats)
        return pooled


# ────────────────────────────────────────────────────────────────────────────────
# IoT Transformer Encoder (Backbone)
# ────────────────────────────────────────────────────────────────────────────────
class IoTTransformerEncoder(nn.Module):
    """
    Transformer Encoder backbone for IoT flow sequences.

    Takes a batch of flow windows (B, T, input_dim) and produces:
      - Sequence of hidden states (B, T, d_model)  — for MLM/seq2seq tasks
      - Pooled embedding vector (B, d_model)        — for contrastive/classification

    This is the "xương sống" (backbone) of the system, adapted from ET-BERT's
    Transformer Encoder with Sinusoidal Positional Encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedfwd: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_feedfwd = dim_feedfwd or 4 * d_model
        self.d_model = d_model

        # 1) Feature → d_model projection (Linear Embedding)
        # Replaces ET-BERT's WordPiece tokenizer with a linear projection
        # for numeric CSV features
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 2) Positional encoding for flow ordering
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        # 3) Transformer Encoder stack (batch-first, pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # 4) Attention pooling for sequence → vector
        self.pool = MaskedAttentionPooling(d_model)

    def _make_key_padding_mask(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """True for PAD positions — shape (B, max_len)."""
        return (
            torch.arange(max_len, device=lengths.device).expand(lengths.size(0), -1)
            >= lengths.unsqueeze(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ):
        """
        Parameters
        ----------
        x : (B, T, input_dim) — batch of flow windows
        lengths : (B,) — actual lengths (if None, assumes all T)
        return_sequence : bool — if True, return (B, T, d_model) instead of pooled

        Returns
        -------
        If return_sequence:
            hidden_states : (B, T, d_model)
        Else:
            embedding : (B, d_model) — pooled representation
        """
        B, T, _ = x.shape

        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        # Project + positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        # Key padding mask
        key_padding_mask = self._make_key_padding_mask(lengths, T)

        # Encode
        hidden = self.encoder(h, src_key_padding_mask=key_padding_mask)

        if return_sequence:
            return hidden

        # Pool to single vector
        embedding = self.pool(hidden, key_padding_mask)
        return embedding


# ────────────────────────────────────────────────────────────────────────────────
# IoT Device Classifier (Fine-tuning head)
# ────────────────────────────────────────────────────────────────────────────────
class IoTDeviceClassifier(nn.Module):
    """
    Full model: Transformer Encoder + Classification head.
    Used during fine-tuning phase to classify IoT devices.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedfwd: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = IoTTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedfwd=dim_feedfwd,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : (B, T, input_dim)
        lengths : (B,) or None

        Returns
        -------
        logits : (B, num_classes)
        """
        embedding = self.encoder(x, lengths, return_sequence=False)
        logits = self.classifier(embedding)
        return logits

    def get_embeddings(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """Extract embeddings without classification (for t-SNE, etc.)."""
        return self.encoder(x, lengths, return_sequence=False)


# ────────────────────────────────────────────────────────────────────────────────
# Masked Feature Modeling Head (ET-BERT-style Pre-training)
# ────────────────────────────────────────────────────────────────────────────────
class MaskedFeatureModeling(nn.Module):
    """
    ET-BERT-style pre-training: Mask random features/flows in a window
    and train the Transformer to reconstruct them.

    Instead of masking bytes in raw packets (ET-BERT original), we mask
    entire feature values in CSV flow records. The model learns to predict
    srcAvgPayloadSize from surrounding flow context — understanding deep
    behavioral patterns without any labels.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedfwd: Optional[int] = None,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.input_dim = input_dim

        self.encoder = IoTTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedfwd=dim_feedfwd,
            dropout=dropout,
        )

        # Reconstruction head: predict original feature values
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(input_dim) * 0.02)

    def create_mask(self, x: torch.Tensor) -> Tuple:
        """
        Create random masks for flow features.

        Strategy: For each flow in the window, randomly mask `mask_ratio`
        of the features (set them to the learned mask token).

        Returns
        -------
        x_masked : (B, T, input_dim) — input with masked features
        mask : (B, T, input_dim) — boolean mask (True = masked)
        """
        B, T, F = x.shape
        mask = torch.rand(B, T, F, device=x.device) < self.mask_ratio

        x_masked = x.clone()
        x_masked[mask] = self.mask_token.expand(B, T, -1)[mask]

        return x_masked, mask

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : (B, T, input_dim) — original (unmasked) flow window

        Returns
        -------
        loss : scalar — MSE reconstruction loss on masked positions
        reconstructed : (B, T, input_dim) — full reconstruction
        """
        # Create masks and corrupt input
        x_masked, mask = self.create_mask(x)

        # Encode masked input
        hidden = self.encoder(x_masked, lengths, return_sequence=True)

        # Reconstruct
        reconstructed = self.reconstruction_head(hidden)

        # Loss only on masked positions
        if mask.any():
            loss = F.mse_loss(reconstructed[mask], x[mask])
        else:
            loss = torch.tensor(0.0, device=x.device)

        return loss, reconstructed


# ────────────────────────────────────────────────────────────────────────────────
# Collate function for DataLoader
# ────────────────────────────────────────────────────────────────────────────────
def iot_collate_fn(batch):
    """
    Collate for IoTFlowWindowDataset.
    All windows have the same size, so simple stacking works.
    """
    features, labels = zip(*batch)
    features = torch.stack(features)  # (B, T, F)
    labels = torch.stack(labels)      # (B,)
    return features, labels


def iot_contrastive_collate_fn(batch):
    """
    Collate for IoTContrastiveDataset.
    Returns anchors, positives, negatives, labels.
    """
    anchors, positives, negatives, labels = zip(*batch)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
        torch.stack(labels),
    )


# ────────────────────────────────────────────────────────────────────────────────
# Quick sanity test
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== IoT Transformer Sanity Test ===\n")

    B, T, D = 4, 10, 17  # 17 UNSW features
    x = torch.randn(B, T, D)

    # Test encoder backbone
    encoder = IoTTransformerEncoder(input_dim=D, d_model=64, nhead=4, num_layers=2)
    emb = encoder(x)
    print(f"Encoder embedding shape: {emb.shape}")  # (4, 64)

    seq = encoder(x, return_sequence=True)
    print(f"Encoder sequence shape:  {seq.shape}")  # (4, 10, 64)

    # Test classifier
    classifier = IoTDeviceClassifier(input_dim=D, num_classes=27, d_model=64, nhead=4, num_layers=2)
    logits = classifier(x)
    print(f"Classifier logits shape: {logits.shape}")  # (4, 27)

    # Test MLM pre-training
    mlm = MaskedFeatureModeling(input_dim=D, d_model=64, nhead=4, num_layers=2)
    loss, recon = mlm(x)
    print(f"MLM loss: {loss.item():.4f}, reconstruction shape: {recon.shape}")

    print("\n[OK] All models pass sanity check!")

