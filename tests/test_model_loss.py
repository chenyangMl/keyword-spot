import torch

from model_convert.model.loss import (
    max_pooling_loss, acc_frame, ctc_loss, cross_entropy,
    criterion, ctc_prefix_beam_search, Calculator
)


class TestAccFrame:
    def test_perfect_prediction(self):
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.1, 0.9]])
        target = torch.tensor([1, 0, 1])
        acc = acc_frame(logits, target)
        assert acc == 100.0

    def test_no_correct(self):
        logits = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        target = torch.tensor([1, 0])
        acc = acc_frame(logits, target)
        assert acc == 0.0

    def test_none_logits(self):
        acc = acc_frame(None, torch.tensor([1, 0]))
        assert acc == 0


class TestMaxPoolingLoss:
    def test_returns_loss_and_acc(self):
        B, T, D = 2, 5, 2
        logits = torch.sigmoid(torch.randn(B, T, D))
        target = torch.tensor([0, 1])
        lengths = torch.tensor([5, 5], dtype=torch.int32)
        loss, acc = max_pooling_loss(logits, target, lengths)
        assert isinstance(loss.item(), float)
        assert 0.0 <= acc <= 1.0

    def test_perfect_keyword(self):
        # keyword 0 active in utt 0
        logits = torch.zeros(1, 3, 2)
        logits[0, :, 0] = 0.99  # high prob for keyword 0
        logits[0, :, 1] = 0.01
        target = torch.tensor([0])
        lengths = torch.tensor([3], dtype=torch.int32)
        loss, acc = max_pooling_loss(logits, target, lengths)
        assert loss.item() < 0.1
        assert acc == 1.0


class TestCrossEntropy:
    def test_returns_loss_and_acc(self):
        logits = torch.randn(4, 3)
        target = torch.tensor([0, 1, 2, 1])
        loss, acc = cross_entropy(logits, target)
        assert loss.item() > 0
        assert 0.0 <= acc <= 100.0

    def test_perfect_logits(self):
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        target = torch.tensor([0, 1])
        loss, acc = cross_entropy(logits, target)
        assert acc == 100.0


class TestCtcLoss:
    def test_returns_loss_and_acc(self):
        B, T, D = 2, 20, 5
        logits = torch.randn(B, T, D)
        target = torch.tensor([[1, 2], [3, 4]])
        logits_lengths = torch.tensor([20, 20])
        target_lengths = torch.tensor([2, 2])
        loss, acc = ctc_loss(logits, target, logits_lengths, target_lengths)
        assert loss.item() > 0


class TestCriterion:
    def test_ce(self):
        logits = torch.randn(4, 3)
        target = torch.tensor([0, 1, 2, 1])
        lengths = torch.tensor([4, 4, 4, 4])
        loss, acc = criterion('ce', logits, target, lengths)
        assert loss.item() > 0

    def test_max_pooling(self):
        logits = torch.sigmoid(torch.randn(2, 5, 2))
        target = torch.tensor([0, 1])
        lengths = torch.tensor([5, 5], dtype=torch.int32)
        loss, acc = criterion('max_pooling', logits, target, lengths)
        assert isinstance(loss.item(), float)

    def test_ctc(self):
        logits = torch.randn(2, 20, 5)
        target = torch.tensor([[1, 2], [3, 4]])
        logits_lengths = torch.tensor([20, 20])
        target_lengths = torch.tensor([2, 2])
        loss, acc = criterion('ctc', logits, target, logits_lengths,
                              target_lengths=target_lengths)
        assert loss.item() > 0


class TestCtcPrefixBeamSearch:
    def test_basic_search(self):
        T, D = 10, 4
        logits = torch.softmax(torch.randn(T, D), dim=-1)
        logits_lengths = torch.tensor(T)
        hyps = ctc_prefix_beam_search(logits, logits_lengths, None, 3, 5)
        assert isinstance(hyps, list)
        # Each hyp is (prefix_tuple, score, nodes)
        for hyp in hyps:
            assert len(hyp) == 3

    def test_with_keyword_filter(self):
        T, D = 10, 4
        logits = torch.softmax(torch.randn(T, D), dim=-1)
        logits_lengths = torch.tensor(T)
        keywords = {1, 2}
        hyps = ctc_prefix_beam_search(logits, logits_lengths, keywords, 3, 5)
        assert isinstance(hyps, list)


class TestCalculator:
    def test_identical_sequences(self):
        calc = Calculator()
        lab = ['a', 'b', 'c']
        rec = ['a', 'b', 'c']
        result = calc.calculate(lab, rec)
        assert result['all'] == 3
        assert result['cor'] == 3
        assert result['sub'] == 0
        assert result['ins'] == 0
        assert result['del'] == 0

    def test_substitution(self):
        calc = Calculator()
        lab = ['a', 'b']
        rec = ['a', 'x']
        result = calc.calculate(lab, rec)
        assert result['sub'] == 1

    def test_deletion(self):
        calc = Calculator()
        lab = ['a', 'b', 'c']
        rec = ['a', 'c']
        result = calc.calculate(lab, rec)
        assert result['del'] == 1

    def test_insertion(self):
        calc = Calculator()
        lab = ['a', 'c']
        rec = ['a', 'b', 'c']
        result = calc.calculate(lab, rec)
        assert result['ins'] == 1

    def test_overall(self):
        calc = Calculator()
        calc.calculate(['a', 'b'], ['a', 'b'])
        calc.calculate(['c'], ['d'])
        overall = calc.overall()
        assert overall['all'] == 3
        assert overall['cor'] == 2
        assert overall['sub'] == 1

    def test_keys(self):
        calc = Calculator()
        calc.calculate(['a', 'b'], ['a', 'c'])
        keys = calc.keys()
        assert 'a' in keys
        assert 'b' in keys
        assert 'c' in keys

    def test_cluster(self):
        calc = Calculator()
        calc.calculate(['a', 'b'], ['a', 'b'])
        result = calc.cluster(['a'])
        assert result['all'] == 1
        assert result['cor'] == 1
