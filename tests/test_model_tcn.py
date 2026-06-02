import torch

from model_convert.model.tcn import CnnBlock, DsCnnBlock, TCN


class TestCnnBlock:
    def test_output_shape(self):
        block = CnnBlock(channel=16, kernel_size=3, dilation=1)
        x = torch.randn(2, 16, 10)  # (B, D, T)
        out, cache = block(x)
        assert out.shape == (2, 16, 10)

    def test_with_cache(self):
        block = CnnBlock(channel=16, kernel_size=3, dilation=2)
        x = torch.randn(2, 16, 5)
        out1, cache = block(x)
        # Use cache for next chunk
        x2 = torch.randn(2, 16, 5)
        out2, cache2 = block(x2, cache)
        assert out2.shape == (2, 16, 5)
        assert cache2.shape[2] == block.padding


class TestDsCnnBlock:
    def test_output_shape(self):
        block = DsCnnBlock(channel=16, kernel_size=3, dilation=1)
        x = torch.randn(2, 16, 10)
        out, cache = block(x)
        assert out.shape == (2, 16, 10)

    def test_with_cache(self):
        block = DsCnnBlock(channel=16, kernel_size=3, dilation=2)
        x = torch.randn(2, 16, 5)
        out1, cache = block(x)
        x2 = torch.randn(2, 16, 5)
        out2, cache2 = block(x2, cache)
        assert out2.shape == (2, 16, 5)


class TestTCN:
    def test_output_shape(self):
        tcn = TCN(num_layers=3, channel=32, kernel_size=4, dropout=0.0,
                  block_class=CnnBlock)
        x = torch.randn(2, 10, 32)  # (B, T, D)
        out, cache = tcn(x)
        assert out.shape == (2, 10, 32)

    def test_cache_shape(self):
        tcn = TCN(num_layers=3, channel=32, kernel_size=4, dropout=0.0,
                  block_class=CnnBlock)
        x = torch.randn(2, 10, 32)
        out, cache = tcn(x)
        assert cache.shape == (2, 32, tcn.padding)

    def test_with_cache(self):
        tcn = TCN(num_layers=2, channel=16, kernel_size=3, dropout=0.0,
                  block_class=CnnBlock)
        tcn.eval()
        x1 = torch.randn(1, 5, 16)
        out1, cache1 = tcn(x1)
        x2 = torch.randn(1, 5, 16)
        out2, cache2 = tcn(x2, cache1)
        assert out2.shape == (1, 5, 16)

    def test_ds_cnn_block(self):
        tcn = TCN(num_layers=2, channel=16, kernel_size=3, dropout=0.0,
                  block_class=DsCnnBlock)
        x = torch.randn(2, 10, 16)
        out, cache = tcn(x)
        assert out.shape == (2, 10, 16)

    def test_padding_calculation(self):
        tcn = TCN(num_layers=3, channel=16, kernel_size=4, dropout=0.0)
        # dilation = 2^i, padding = (kernel-1)*dilation
        # layer 0: (4-1)*1=3, layer 1: (4-1)*2=6, layer 2: (4-1)*4=12
        assert tcn.padding == 3 + 6 + 12
