import numpy as np
import pandas as pd

from toxic_comments.evaluation import calculate_metrics, select_threshold


def test_select_threshold_and_metrics_for_separable_scores() -> None:
    labels = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    threshold = select_threshold(labels, scores)
    metrics = calculate_metrics(labels, scores, threshold)

    assert 0.2 < threshold <= 0.8
    assert metrics.f1_toxic == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.roc_auc == 1.0
