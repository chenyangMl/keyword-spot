import torch

from model_convert.model.mdtc import DSDilatedConv1d, TCNBlock, TCNStack, MDTC


class TestDSDilatedConv1d:
    def test_output_shape(self):
        conv = DSDilatedConv1d(16, 32, kernel_size=3, dilation=1)
        # input needs padding handled externally; raw input (B, C, T)
        x = torch.randn(2, 16, 10)
        out = conv(x)
        # output T depends on padding=dilation*(k-1)=1*2=2, no padding added
        # Conv1d with no padding: T_out = T - dilation*(k-1) = 10 - 2 = 8
        assert out.shape == (2, 32, 8)

    def test_dilation_effect(self):
        conv = DSDilatedConv1d(16, 16, kernel_size=3, dilation=2)
        x = torch.randn(1, 16, 20)
        out = conv(x)
        # T_out = 20 - 2*(3-1) = 16
        assert out.shape == (1, 16, 16)


class TestTCNBlock:
    def test_output_shape(self):
        block = TCNBlock(in_channels=32, res_channels=32,
                         kernel_size=3, dilation=1, causal=True)
        x = torch.randn(2, 32, 10)
        out, cache = block(x)
        assert out.shape == (2, 32, 10)

    def test_cache_shape(self):
        block = TCNBlock(in_channels=32, res_channels=32,
                         kernel_size=3, dilation=2, causal=True)
        x = torch.randn(2, 32, 10)
        out, cache = block(x)
        assert cache.shape == (2, 32, block.padding)

    def test_with_cache(self):
        block = TCNBlock(in_channels=16, res_channels=16,
                         kernel_size=3, dilation=1, causal=True)
        block.eval()
        x1 = torch.randn(1, 16, 5)
        out1, cache1 = block(x1)
        x2 = torch.randn(1, 16, 5)
        out2, cache2 = block(x2, cache1)
        assert out2.shape == (1, 16, 5)


class TestTCNStack:
    def test_output_shape(self):
        stack = TCNStack(in_channels=32, stack_num=2, stack_size=1,
                         res_channels=32, kernel_size=3, causal=True)
        x = torch.randn(2, 32, 10)
        out, cache = stack(x)
        assert out.shape == (2, 32, 10)

    def test_padding(self):
        stack = TCNStack(in_channels=32, stack_num=2, stack_size=1,
                         res_channels=32, kernel_size=3, causal=True)
        assert stack.padding > 0


class TestMDTC:
    def test_output_shape(self):
        mdtc = MDTC(stack_num=2, stack_size=2, in_channels=32,
                     res_channels=32, kernel_size=3, causal=True)
        x = torch.randn(2, 10, 32)  # (B, T, D)
        out, cache = mdtc(x)
        assert out.shape == (2, 10, 32)

    def test_with_cache(self):
        mdtc = MDTC(stack_num=1, stack_size=2, in_channels=16,
                     res_channels=16, kernel_size=3, causal=True)
        mdtc.eval()
        x1 = torch.randn(1, 5, 16)
        out1, cache1 = mdtc(x1)
        x2 = torch.randn(1, 5, 16)
        out2, cache2 = mdtc(x2, cache1)
        assert out2.shape == (1, 5, 16)

    def test_cache_accumulates(self):
        mdtc = MDTC(stack_num=2, stack_size=2, in_channels=16,
                     res_channels=16, kernel_size=3, causal=True)
        x = torch.randn(1, 10, 16)
        out, cache = mdtc(x)
        assert cache.shape[2] == mdtc.padding
