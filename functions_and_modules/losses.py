import torch
import torch.nn as nn
import torch.nn.functional as F


class CRCLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        device = features.device

        labels = labels.contiguous().view(-1, 1)

        # Mask: same label = positive pair
        mask_positive = torch.eq(labels, labels.T).float()  # (B, B)

        # Compute scaled cosine similarity
        logits = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Remove self-similarity from diagonal
        logits_mask = torch.ones(batch_size, batch_size, device=device) - torch.eye(
            batch_size, device=device
        )
        logits = logits * logits_mask

        # For each sample, compute CRC loss
        # Positive pairs: same device, different sample
        # Negative pairs: different device
        pos_mask = mask_positive * logits_mask  # positive pairs (excluding self)
        neg_mask = (1.0 - mask_positive) * logits_mask  # negative pairs

        # Numerator: exp(sim) for positive pairs
        pos_logits = logits * pos_mask
        # Denominator: sum of exp(sim) for ALL negative pairs (CRC: global sum)
        neg_sum = torch.sum(torch.exp(logits * neg_mask))

        # For each positive pair, compute log-softmax
        # CRC uses global negative sum instead of per-row (InfoNCE)
        num_positives = pos_mask.sum(dim=1).clamp(min=1)

        # Log probability for positive pairs
        log_prob = pos_logits - torch.log(
            torch.exp(pos_logits) + neg_sum + 1e-8
        )
        log_prob = (log_prob * pos_mask).sum(dim=1) / num_positives

        loss = -log_prob.mean() * self.temperature
        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (B,)

        # Negative similarity
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature  # (B,)

        # NT-Xent: -log(exp(pos) / (exp(pos) + exp(neg)))
        logits = torch.stack([pos_sim, neg_sim], dim=1)  # (B, 2)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        return self.loss_fn(anchor, positive, negative)


class CombinedSSLLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.contrastive = NTXentLoss(temperature)

    def forward(
        self,
        recon_loss: torch.Tensor,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        contrastive_loss = self.contrastive(anchor_emb, positive_emb, negative_emb)
        total = self.alpha * recon_loss + self.beta * contrastive_loss
        return total, recon_loss, contrastive_loss
