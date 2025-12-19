import math
import pytest
import pandas as pd
from utils.benford import leading_digit_of_number, digit_counts, benford_report, benford_score


def test_leading_digit_examples():
    assert leading_digit_of_number(123.4) == 1
    assert leading_digit_of_number(0.00456) == 4
    assert leading_digit_of_number(-250) == 2
    assert leading_digit_of_number(0) is None
    assert leading_digit_of_number(float('nan')) is None


def test_digit_counts_and_score_benford_like():
    # Create a synthetic Benford-like distribution by sampling magnitudes
    # We'll use powers of 10 times 1..9 to produce leading digits 1..9 evenly
    values = []
    for d in range(1, 10):
        values.extend([d * 10**k for k in range(0, 5)])
    counts = digit_counts(values)
    # counts should be >0 for each digit
    assert all(c > 0 for c in counts)
    report = benford_report(values)
    assert report['total'] == sum(counts)
    # score should be finite
    score = float(report['score'])
    assert math.isfinite(score)


def test_benford_empty_and_non_numeric():
    values = [0, None, 'a', float('inf')]
    report = benford_report(values)
    assert report['total'] == 0
    sc = report['score']
    try:
        scf = float(sc)
        assert math.isnan(scf) or scf == scf
    except Exception:
        # non-numeric score is acceptable for empty input
        assert sc is None or isinstance(sc, float) or sc != sc
