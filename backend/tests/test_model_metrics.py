import numpy as np
import pytest
import torch

from app.evaluation import metrics as M
from app.models.cnn import ECGNet, count_parameters, describe_architecture
from app.training.trainer import class_weights


def test_forward_shapes():
    m = ECGNet(n_classes=4)
    out = m(torch.randn(8, 1, 256), torch.randn(8, 4))
    assert out.shape == (8, 4)
    m2 = ECGNet(n_classes=5, use_rr=False)
    assert m2(torch.randn(3, 1, 256)).shape == (3, 5)
    with pytest.raises(ValueError):
        m(torch.randn(2, 1, 256))


def test_architecture_is_a_real_1d_cnn():
    m = ECGNet()
    convs = [x for x in m.modules() if isinstance(x, torch.nn.Conv1d)]
    assert len(convs) == 7
    assert 50_000 < count_parameters(m) < 1_000_000
    rows = describe_architecture(m, 256)
    assert rows[-1]["output_shape"] == [4]


def test_seeded_initialisation_is_reproducible():
    torch.manual_seed(7)
    a = ECGNet()
    torch.manual_seed(7)
    b = ECGNet()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    x = torch.randn(32, 1, 256)
    rr = torch.randn(32, 4)
    y = torch.arange(32) % 4
    m = ECGNet(dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x, rr), y)
        loss.backward()
        opt.step()
    m.eval()
    assert (m(x, rr).argmax(1) == y).float().mean() > 0.95


def test_metrics_on_known_confusion():
    y = np.array([0] * 90 + [1] * 10)
    p = np.array([0] * 85 + [1] * 5 + [1] * 6 + [0] * 4)
    r = M.compute(y, p, ["N", "S"])
    assert r["confusion_matrix"] == [[85, 5], [4, 6]]
    assert r["accuracy"] == pytest.approx(0.91)
    assert r["per_class"]["S"]["recall"] == pytest.approx(0.6)
    assert r["per_class"]["S"]["precision"] == pytest.approx(6 / 11, abs=1e-4)
    assert r["per_class"]["S"]["specificity"] == pytest.approx(85 / 90, abs=1e-4)


def test_majority_classifier_shows_accuracy_trap():
    y = np.array([0] * 89 + [1] * 5 + [2] * 6)
    r = M.compute(y, np.zeros_like(y), ["N", "S", "V"])
    assert r["accuracy"] == pytest.approx(0.89)
    assert r["macro_f1"] < 0.35


def test_bootstrap_resamples_subjects():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = y.copy()
    p[:40] = 1 - p[:40]
    subj = np.repeat(np.arange(8), 50)
    ci = M.bootstrap_ci(y, p, subj, ["N", "S"], n_boot=100)
    lo, hi = ci["accuracy"]
    assert lo <= 0.9 <= hi
    assert ci["unit"] == "subject"


def test_class_weights_inverse_frequency():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y, 2, power=1.0)
    assert w[1] / w[0] == pytest.approx(9.0)
