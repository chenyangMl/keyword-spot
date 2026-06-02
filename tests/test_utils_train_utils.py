import torch
import torch.nn as nn

from model_convert.utils.train_utils import set_mannul_seed, count_parameters


class TestSetMannulSeed:
    def test_reproducibility(self):
        set_mannul_seed(42)
        a = torch.randn(3, 3)
        set_mannul_seed(42)
        b = torch.randn(3, 3)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_mannul_seed(1)
        a = torch.randn(5)
        set_mannul_seed(2)
        b = torch.randn(5)
        assert not torch.equal(a, b)


class TestCountParameters:
    def test_linear_layer(self):
        model = nn.Linear(10, 5)
        # weight: 10*5=50, bias: 5 => 55
        assert count_parameters(model) == 55

    def test_frozen_params_not_counted(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        assert count_parameters(model) == 0

    def test_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(4, 3, bias=False),  # 12
            nn.Linear(3, 2, bias=False),  # 6
        )
        assert count_parameters(model) == 18
