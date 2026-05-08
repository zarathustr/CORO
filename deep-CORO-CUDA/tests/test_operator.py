import torch

from deep_coro_cuda.ops import deep_coro_forward, deep_coro_forward_torch, random_rotations


def test_shapes_and_orthogonality():
    torch.manual_seed(0)
    M = random_rotations(32) @ torch.diag_embed(torch.exp(0.3 * torch.randn(32, 3)))
    alpha = torch.ones(8)
    beta = torch.ones(8)
    R = deep_coro_forward_torch(M, alpha, beta)
    assert R.shape == (32, 3, 3)
    I = torch.eye(3)
    err = torch.linalg.norm(R.transpose(-1, -2) @ R - I, dim=(-2, -1)).mean().item()
    assert err < 1e-4


def test_extension_or_fallback_matches_torch():
    torch.manual_seed(1)
    M = random_rotations(64) @ torch.diag_embed(torch.exp(0.3 * torch.randn(64, 3)))
    alpha = torch.ones(4)
    beta = torch.ones(4)
    R0 = deep_coro_forward_torch(M, alpha, beta)
    R1 = deep_coro_forward(M, alpha, beta, use_extension=True)
    assert torch.max(torch.abs(R0 - R1)).item() < 1e-5
