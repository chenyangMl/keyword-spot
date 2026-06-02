import torch
import torch.nn as nn

from model_convert.model.classifier import (
    GlobalClassifier, LastClassifier, ElementClassifier, LinearClassifier
)


class TestGlobalClassifier:
    def test_output_shape(self):
        linear = nn.Linear(16, 3)
        clf = GlobalClassifier(linear)
        x = torch.randn(2, 10, 16)  # (B, T, D)
        out = clf(x)
        assert out.shape == (2, 3)

    def test_mean_pooling(self):
        linear = nn.Identity()
        clf = GlobalClassifier(linear)
        x = torch.ones(1, 5, 4)
        out = clf(x)
        assert torch.allclose(out, torch.ones(1, 4))


class TestLastClassifier:
    def test_output_shape(self):
        linear = nn.Linear(16, 3)
        clf = LastClassifier(linear)
        x = torch.randn(2, 10, 16)
        out = clf(x)
        assert out.shape == (2, 3)

    def test_selects_last_frame(self):
        linear = nn.Identity()
        clf = LastClassifier(linear)
        x = torch.zeros(1, 5, 4)
        x[0, -1, :] = 1.0
        out = clf(x)
        assert torch.allclose(out, torch.ones(1, 4))


class TestElementClassifier:
    def test_output_shape(self):
        linear = nn.Linear(16, 3)
        clf = ElementClassifier(linear)
        x = torch.randn(2, 10, 16)
        out = clf(x)
        assert out.shape == (2, 10, 3)


class TestLinearClassifier:
    def test_output_shape(self):
        clf = LinearClassifier(8, 4)
        x = torch.randn(2, 10, 8)
        out = clf(x)
        assert out.shape == (2, 10, 4)

    def test_deterministic(self):
        clf = LinearClassifier(8, 4)
        clf.eval()
        x = torch.randn(1, 5, 8)
        out1 = clf(x)
        out2 = clf(x)
        assert torch.allclose(out1, out2)
