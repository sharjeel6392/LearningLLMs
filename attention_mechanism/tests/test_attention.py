import torch
import pytest
from src import SelfAttention_v1
from src import SelfAttention_v2

def test_forward_output_shape():
    """Check that forward pass returns the right shape"""
    seq_len, d_in, d_out = 3, 4, 5
    model = SelfAttention_v1(d_in, d_out)
    x = torch.rand(seq_len, d_in)  # input
    out = model(x)
    assert out.shape == (seq_len, d_out)


def test_forward_dtype():
    """Check that output has same dtype as input"""
    model = SelfAttention_v1(4, 6)
    x = torch.rand(3, 4, dtype=torch.float32)
    out = model(x)

    assert out.dtype == torch.float32


def test_grad_flow():
    """Check that gradients flow through parameters"""
    model = SelfAttention_v1(4, 6)
    x = torch.rand(3, 4, dtype=torch.float32, requires_grad=True)

    out = model(x).sum()
    out.backward()

    # ensure at least one parameter got gradient
    grads = [p.grad for p in model.parameters()]
    assert any(g is not None for g in grads)

def test_v1_from_book():
    """Test the v1 example from the book"""
    x = torch.tensor(
        [
            [0.43, 0.15, 0.89], # Your     (x^1)
            [0.55, 0.87, 0.66], # journey  (x^2)
            [0.57, 0.85, 0.64], # starts   (x^3)
            [0.22, 0.58, 0.33], # with     (x^4)
            [0.77, 0.25, 0.10], # one      (x^5)
            [0.05, 0.80, 0.55] # step      (x^6)
        ]
    )

    out_expected = torch.tensor(
        [
            [0.2996, 0.8053],
            [0.3061, 0.8210],
            [0.3058, 0.8203],
            [0.2948, 0.7939],
            [0.2927, 0.7891],
            [0.2990, 0.8040]
        ]
    )
    torch.manual_seed(123)
    d_in, d_out = x.shape[1], 2
    sa_v1 = SelfAttention_v1(d_in, d_out)
    out = sa_v1.forward(x)
    
    torch.testing.assert_close(out, out_expected, atol = 1e-4, rtol = 0)


def test_v2_from_book():
    """Test the v2 example from the book"""
    x = torch.tensor(
        [
            [0.43, 0.15, 0.89], # Your     (x^1)
            [0.55, 0.87, 0.66], # journey  (x^2)
            [0.57, 0.85, 0.64], # starts   (x^3)
            [0.22, 0.58, 0.33], # with     (x^4)
            [0.77, 0.25, 0.10], # one      (x^5)
            [0.05, 0.80, 0.55] # step      (x^6)
        ]
    )

    out_expected = torch.tensor(
        [
            [-0.0739,  0.0713],
            [-0.0748,  0.0703],
            [-0.0749,  0.0702],
            [-0.0760,  0.0685],
            [-0.0763,  0.0679],
            [-0.0754,  0.0693]
        ]
    )
    torch.manual_seed(789)
    d_in, d_out = x.shape[1], 2
    sa_v1 = SelfAttention_v2(d_in, d_out)
    out = sa_v1.forward(x)
    torch.testing.assert_close(out, out_expected, atol = 1e-4, rtol = 0)