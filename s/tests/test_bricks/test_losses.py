import torch

from sprite.bricks.losses import BCEFocalLoss, ContrastiveLoss, MarginLoss


def test_loss():
    loss_function = MarginLoss(1)
    inputs = torch.tensor([0.1, 0.2])
    label = torch.tensor([1, 2])
    result = loss_function(inputs, label)
    assert type(result) == torch.Tensor
    assert abs(result.cpu().detach().item() - 5.72) < 0.01

    loss_function = ContrastiveLoss(1)
    inputs1 = torch.tensor([[0.05, 0.6], [-0.7, 0.9]])
    inputs2 = torch.tensor([[0.3, -0.15], [-0.1, 0.24]])
    label = torch.tensor([1, 0])
    result = loss_function(inputs1, inputs2, label)
    assert type(result) == torch.Tensor
    assert abs(result.cpu().detach().item() - 1.4057) < 0.01

    loss_function = BCEFocalLoss(0.5, 2)
    inputs = torch.tensor([0.1, 0.2, -0.3, -0.7])
    label = torch.tensor([1, 0, 0, 1])
    result = loss_function(inputs, label)
    assert type(result) == torch.Tensor
    assert abs(result.cpu().detach().item() - 0.1225) < 0.01
