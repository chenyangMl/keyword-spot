import torch

from model_convert.model.subsampling import (
    NoSubsampling, LinearSubsampling1, Conv1dSubsampling1
)


class TestNoSubsampling:
    def test_passthrough(self):
        m = NoSubsampling()
        x = torch.randn(2, 10, 4)
        out = m(x)
        assert torch.equal(out, x)

    def test_subsampling_rate(self):
        m = NoSubsampling()
        assert m.subsampling_rate == 1


class TestLinearSubsampling1:
    def test_output_shape(self):
        m = LinearSubsampling1(40, 64)
        x = torch.randn(2, 10, 40)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_subsampling_rate(self):
        m = LinearSubsampling1(40, 64)
        assert m.subsampling_rate == 1

    def test_deterministic_in_eval(self):
        m = LinearSubsampling1(20, 32)
        m.eval()
        x = torch.randn(1, 5, 20)
        out1 = m(x)
        out2 = m(x)
        assert torch.allclose(out1, out2)


class TestConv1dSubsampling1:
    def test_output_shape(self):
        m = Conv1dSubsampling1(40, 64)
        # Conv1d input: (B, idim, T), kernel=3, no padding => output T' = T - 2
        x = torch.randn(2, 40, 10)
        out = m(x)
        assert out.shape == (2, 64, 8)

    def test_subsampling_rate(self):
        m = Conv1dSubsampling1(40, 64)
        assert m.subsampling_rate == 1
