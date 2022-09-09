import torch
from torch import nn
from torch.nn.functional import cosine_similarity, pairwise_distance


class MarginLoss(nn.Module):
    """Margin loss function.

    Attributes:
        margin (float, optional):
            The margin for clamping loss.
            Defaults to 1.0.
        reduction (str,optional):
            Ways for reduction.
            Defaults to "mean".

    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward function of MarginLoss.

        Args:
            x1 (torch.Tensor):
                The positive score tensor. shape:(b,)
            x2 (torch.Tensor):
                The negtive score tensor. shape:(b,)

        Returns:
            torch.Tensor:
                MarginLoss result. shape:(b,)
        """
        loss = torch.pow(torch.clamp(x2 - x1 + self.margin, min=0.0), 2)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive Loss function.

    Attributes:
        margin (float, optional):
            The margin for clamping loss.
            Defaults to 1.0.
        distance (str,optional):
            Distance for getting similarity.
            Defaults to "cosine".
        reduction (str,optional):
            Ways for reduction.
            Defaults to "mean".

    """

    def __init__(self, margin=1.0, distance="cosine", reduction: str = "mean") -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        if distance not in ("cosine", "euclidean"):
            raise Exception("invalid distance")
        self.distance = distance
        if self.distance == "cosine":
            self._distance_func = lambda x1, x2: 1.0 - cosine_similarity(x1, x2)
        else:
            self._distance_func = lambda x1, x2: pairwise_distance(x1, x2)

        self.reduction = reduction

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward function of Contrastive.

        Args:
            x1 (torch.Tensor):
                The positive tensor. shape:(b,d)
            x2 (torch.Tensor):
                The negtive tensor. shape:(b,d)
            y (torch.Tensor):
                The label tensor. shape:(b,)

        Returns:
            torch.Tensor:
                Contrastive result. shape:(b,)
        """

        d = self._distance_func(x1, x2)
        loss = y * torch.pow(d, 2) + (1 - y) * torch.pow(
            torch.clamp(self.margin - d, min=0.0), 2
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class BCEFocalLoss(nn.Module):
    """BCEFocalLoss function.

    Attributes:
        alpha (float, optional):
            Larger alpha gives more weight to positive samples .
            Defaults to 0.5.
        gamma (float, optional):
            Larger gamma gives more weight to hard samples .
            Defaults to 2.0.
        reduction (str,optional):
            Ways for reduction.
            Defaults to "mean".

    """

    def __init__(
        self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"
    ) -> None:
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function of MarginLoss.

        Args:
            pred (torch.Tensor):
                The pred tensor result with no sigmoid. shape:(b,)
            target (torch.Tensor):
                The true label tensor. shape:(b,)

        Returns:
            torch.Tensor:
                BCEFocalLoss result. shape:(b,)
        """
        probs = torch.sigmoid(pred)
        pt = probs.clamp(min=0.0001, max=0.9999)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * target * torch.log(pt) - (
            1 - self.alpha
        ) * pt**self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
