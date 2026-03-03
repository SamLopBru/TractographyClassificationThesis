import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Down-weights well-classified examples so the model focuses on hard,
    misclassified examples. Useful when certain classes are inherently
    harder to distinguish (e.g., overlapping bundles 24-27).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma=0, this is equivalent to CrossEntropyLoss.
    When gamma>0, easy examples (high p_t) get reduced loss.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = 'mean'):
        """
        Args:
            gamma: Focusing parameter. Higher values put more focus on hard examples.
                   gamma=0 is equivalent to CrossEntropyLoss. Typical values: 1.0-3.0.
            label_smoothing: Label smoothing factor (0.0 = disabled).
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs of shape (N, C)
            targets: Ground truth class indices of shape (N,)
        Returns:
            Focal loss value
        """
        # Compute standard cross-entropy (per sample, no reduction)
        ce_loss = F.cross_entropy(
            logits, targets, 
            label_smoothing=self.label_smoothing, 
            reduction='none'
        )
        
        # Compute p_t (probability of the correct class)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    Pulls together embeddings of samples from the same class and pushes apart
    embeddings of samples from different classes in a temperature-scaled 
    cosine similarity space.
    
    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
               https://arxiv.org/abs/2004.11362
    
    Args:
        temperature: Temperature scaling factor (lower = sharper distribution)
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute SupCon loss.
        
        Args:
            features: L2-normalized embeddings of shape (batch_size, projection_dim)
            labels: Class labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Compute pairwise cosine similarity (features are already L2-normalized)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class, excluding self)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        
        # Remove self-similarity from the diagonal
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask
        
        # Number of positives per sample
        num_positives = positive_mask.sum(dim=1)
        
        # For numerical stability, subtract max from sim_matrix
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        # Compute log-sum-exp of all pairs (excluding self)
        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        
        # Compute log-probability of positive pairs
        log_prob = sim_matrix - log_sum_exp
        
        # Only consider samples that have at least one positive pair
        has_positives = num_positives > 0
        
        if has_positives.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Mean of log-probabilities of positive pairs for each anchor
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)
        
        # Loss is negative mean log-probability (averaged over valid anchors)
        loss = -mean_log_prob[has_positives].mean()
        
        return loss


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Projects encoder embeddings into a lower-dimensional, L2-normalized space
    where contrastive loss is computed. Only used during contrastive pre-training.
    
    Architecture: Linear → BatchNorm → ReLU → Linear → L2-normalize
    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings of shape (batch_size, input_dim)
        
        Returns:
            L2-normalized projections of shape (batch_size, output_dim)
        """
        projected = self.net(x)
        return F.normalize(projected, dim=1)


class HybridSupConLoss(nn.Module):
    """
    Hybrid Classification + Supervised Contrastive Loss.
    
    Combines a standard classification loss (CrossEntropy or Focal) with
    Supervised Contrastive Loss as an auxiliary objective:
    
        L_total = L_classification(logits, labels) + λ · L_SupCon(projections, labels)
    
    The SupCon term acts as a regularizer that improves embedding geometry,
    helping the model better separate hard-to-classify classes.
    
    The projection head is a training-time-only side branch — at inference,
    only the encoder + classifier are used.
    
    Args:
        classification_loss: The primary classification loss (CE or Focal)
        supcon_weight: Lambda weight for the SupCon term (default 0.1)
        temperature: SupCon temperature parameter (default 0.07)
    """
    
    def __init__(
        self, 
        classification_loss: nn.Module,
        supcon_weight: float = 0.1, 
        temperature: float = 0.07
    ):
        super().__init__()
        self.classification_loss = classification_loss
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.supcon_weight = supcon_weight
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        projections: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute hybrid loss.
        
        Args:
            logits: Classification logits of shape (N, C)
            labels: Ground truth class indices of shape (N,)
            projections: L2-normalized projected embeddings of shape (N, projection_dim).
                         If None, only classification loss is computed (e.g. during validation).
        
        Returns:
            Scalar loss value
        """
        classification_loss = self.classification_loss(logits, labels)
        
        if projections is not None:
            sc_loss = self.supcon_loss(projections, labels)
            return classification_loss + self.supcon_weight * sc_loss
        
        return classification_loss


def _make_loss(
    loss_name: str,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    supcon_weight: float = 0.1,
    supcon_temperature: float = 0.07
    ) -> nn.Module:
    """
    Factory function for creating loss modules.
    
    Args:
        loss_name: One of 'ce', 'focal', 'supcon', 'hybrid_supcon_ce', 'hybrid_supcon_focal'
        label_smoothing: Label smoothing factor (for CE and Focal)
        focal_gamma: Focusing parameter for FocalLoss
        supcon_weight: Weight of the SupCon term in hybrid losses
        supcon_temperature: Temperature for SupCon similarity scaling
    """
    parts = loss_name.split("_")
    
    if parts[0] == "supcon":
        return SupConLoss(temperature=supcon_temperature)
    elif parts[0] == "focal":
        return FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
    elif parts[0] == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif parts[0] == "hybrid":
        if len(parts) == 3 and parts[1] == "supcon":
            if parts[2] == "ce":
                base = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            elif parts[2] == "focal":
                base = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
            else:
                raise ValueError(f"Unknown loss name: {loss_name}")
            return HybridSupConLoss(
                classification_loss=base,
                supcon_weight=supcon_weight,
                temperature=supcon_temperature
            )
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")