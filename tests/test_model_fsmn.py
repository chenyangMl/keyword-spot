import torch

from model_convert.model.fsmn import (
    toKaldiMatrix, LinearTransform, AffineTransform, FSMNBlock
)


class TestToKaldiMatrix:
    def test_basic(self):
        import numpy as np
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = toKaldiMatrix(mat)
        assert result.startswith('[')
        assert result.strip().endswith(']')
        assert '1.' in result
        assert '4.' in result


class TestLinearTransform:
    def test_output_shape(self):
        lt = LinearTransform(10, 20)
        x = torch.randn(2, 5, 10)
        out, cache = lt(x)
        assert out.shape == (2, 5, 20)

    def test_tuple_input(self):
        lt = LinearTransform(10, 20)
        x = torch.randn(2, 5, 10)
        in_cache = torch.zeros(0, 0, 0, 0)
        out, cache = lt((x, in_cache))
        assert out.shape == (2, 5, 20)

    def test_to_kaldi_net(self):
        lt = LinearTransform(4, 3)
        kaldi_str = lt.to_kaldi_net()
        assert '<LinearTransform> 3 4' in kaldi_str
        assert '<LearnRateCoef> 1' in kaldi_str


class TestAffineTransform:
    def test_output_shape(self):
        at = AffineTransform(10, 20)
        x = torch.randn(2, 5, 10)
        out, cache = at(x)
        assert out.shape == (2, 5, 20)

    def test_tuple_input(self):
        at = AffineTransform(10, 20)
        x = torch.randn(2, 5, 10)
        in_cache = torch.zeros(0, 0, 0, 0)
        out, cache = at((x, in_cache))
        assert out.shape == (2, 5, 20)

    def test_to_kaldi_net(self):
        at = AffineTransform(4, 3)
        kaldi_str = at.to_kaldi_net()
        assert '<AffineTransform> 3 4' in kaldi_str
        assert '<LearnRateCoef>' in kaldi_str


class TestFSMNBlock:
    def test_output_shape(self):
        block = FSMNBlock(input_dim=16, output_dim=16,
                          lorder=5, rorder=2, lstride=1, rstride=1)
        x = torch.randn(2, 10, 16)
        out, cache = block(x)
        assert out.shape == (2, 10, 16)

    def test_with_cache(self):
        block = FSMNBlock(input_dim=16, output_dim=16,
                          lorder=5, rorder=2, lstride=1, rstride=1)
        block.eval()
        x1 = torch.randn(1, 10, 16)
        out1, cache1 = block(x1)
        x2 = torch.randn(1, 10, 16)
        out2, cache2 = block((x2, cache1))
        assert out2.shape == (1, 10, 16)

    def test_no_right_order(self):
        block = FSMNBlock(input_dim=16, output_dim=16,
                          lorder=5, rorder=0, lstride=1, rstride=1)
        assert block.conv_right is None

    def test_to_kaldi_net(self):
        block = FSMNBlock(input_dim=8, output_dim=8,
                          lorder=3, rorder=1, lstride=1, rstride=1)
        kaldi_str = block.to_kaldi_net()
        assert '<Fsmn>' in kaldi_str
        assert '<LOrder> 3' in kaldi_str
        assert '<ROrder> 1' in kaldi_str
