import torch
from model_convert.utils.mask import padding_mask


class TestPaddingMask:
    def test_basic(self):
        lengths = torch.tensor([2, 2, 3], dtype=torch.int32)
        mask = padding_mask(lengths)
        expected = torch.tensor(
            [[False, False, True],
             [False, False, True],
             [False, False, False]]
        )
        assert torch.equal(mask, expected)

    def test_single_element(self):
        lengths = torch.tensor([1], dtype=torch.int32)
        mask = padding_mask(lengths)
        expected = torch.tensor([[False]])
        assert torch.equal(mask, expected)

    def test_all_same_length(self):
        lengths = torch.tensor([3, 3, 3], dtype=torch.int32)
        mask = padding_mask(lengths)
        assert not mask.any()

    def test_varying_lengths(self):
        lengths = torch.tensor([1, 3, 2], dtype=torch.int32)
        mask = padding_mask(lengths)
        expected = torch.tensor(
            [[False, True, True],
             [False, False, False],
             [False, False, True]]
        )
        assert torch.equal(mask, expected)

    def test_output_shape(self):
        lengths = torch.tensor([4, 2, 5], dtype=torch.int32)
        mask = padding_mask(lengths)
        assert mask.shape == (3, 5)

    def test_dtype_is_bool(self):
        lengths = torch.tensor([2, 3], dtype=torch.int32)
        mask = padding_mask(lengths)
        assert mask.dtype == torch.bool
