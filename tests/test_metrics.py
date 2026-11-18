from src.evaluation.metrics import binary_metrics


def test_metrics_perfect():
    m = binary_metrics([0,0,1,1], [.1,.2,.8,.9])
    assert m["roc_auc"] == 1.0
    assert m["specificity"] == 1.0
    assert m["sensitivity"] == 1.0
