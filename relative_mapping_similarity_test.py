import pandas as pd
import numpy as np
import torch
import pytest
from hypothesis import given, strategies as st

from pathlib import Path

from relative_mapping_similarity import normalized_ed, compute_rms


@pytest.fixture
def tables_files():
    dir = Path("rms_test_tables")
    return list(dir.glob("*.csv"))


# nes method needs more testing and it is not a must now.
@given(st.text(), st.text(), st.sampled_from(['max', 'sum']))
def test_normalized_ed(s1, s2, method):
    assert 0 <= normalized_ed(s1, s2, method) <= 1,\
            f"normalized distance should has value within [0, 1] interval."

def test_100_percent_correctness(tables_files):
    table_name = tables_files[2]
    table1 = pd.read_csv(table_name)
    table2 = table1.copy()
    scores = compute_rms(table1, table2)
    for key, value in scores.items():
        if 'similarity' in key or 'rms' in key:
            assert np.isclose(value, 1), f"for table {table_name}:\n{key} should have value of 1, but it is {value}"
        elif 'distance' in key:
            assert np.isclose(value, 0), f"For table {table_name}:\n{key} should have value of 0, but it is {value}"
