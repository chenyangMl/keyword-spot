import torch

from model_convert.model.cmvn import GlobalCMVN


class TestGlobalCMVN:
    def test_output_shape(self):
        mean = torch.zeros(1, 1, 4)
        istd = torch.ones(1, 1, 4)
        cmvn = GlobalCMVN(mean, istd, norm_var=True)
        x = torch.randn(2, 10, 4)
        out = cmvn(x)
        assert out.shape == (2, 10, 4)

    def test_norm_var_true(self):
        mean = torch.tensor([[[1.0, 2.0]]])
        istd = torch.tensor([[[2.0, 0.5]]])
        cmvn = GlobalCMVN(mean, istd, norm_var=True)
        x = torch.tensor([[[3.0, 4.0]]])
        out = cmvn(x)
        # (3-1)*2=4, (4-2)*0.5=1
        expected = torch.tensor([[[4.0, 1.0]]])
        assert torch.allclose(out, expected)

    def test_norm_var_false(self):
        mean = torch.tensor([[[1.0, 2.0]]])
        istd = torch.tensor([[[2.0, 0.5]]])
        cmvn = GlobalCMVN(mean, istd, norm_var=False)
        x = torch.tensor([[[3.0, 4.0]]])
        out = cmvn(x)
        # only subtract mean: (3-1)=2, (4-2)=2
        expected = torch.tensor([[[2.0, 2.0]]])
        assert torch.allclose(out, expected)

    def test_zero_mean_unit_istd_identity(self):
        mean = torch.zeros(1, 1, 3)
        istd = torch.ones(1, 1, 3)
        cmvn = GlobalCMVN(mean, istd, norm_var=True)
        x = torch.randn(2, 5, 3)
        out = cmvn(x)
        assert torch.allclose(out, x)
