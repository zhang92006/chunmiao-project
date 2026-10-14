"""Compare actual scalar/row queries with independent dense source slices."""
from pathlib import Path
import numpy as np
import pytest
from scenario_reconstruction.published_probability_tables import CompactProbabilityTable,ROOT


@pytest.mark.parametrize('path',sorted((ROOT/'source_data/NDD_DATA/LC').glob('*.npz')))
def test_original_probabilities_exact(path):
    table=CompactProbabilityTable(path)
    with np.load(path,allow_pickle=False) as f:
        coordinates=f['coords'];values=f['data'];shape=tuple(f['shape'])
    rng=np.random.default_rng(52)
    for speed_index in (0,10,20):
        # At most 78 MB: never build the three 1.52 GiB dense tensors together.
        dense=np.zeros(shape[1:],dtype=values.dtype)
        selected=coordinates[0]==speed_index
        dense[tuple(coordinates[1:,selected])]=values[selected]
        queries=np.column_stack([rng.integers(size,size=200) for size in shape[1:-1]])
        supported=coordinates[1:-1,selected].T
        if len(supported):
            queries=np.vstack([queries,supported[rng.integers(len(supported),size=200)]])
        for query in queries:
            prefix=(speed_index,)+tuple(query)
            expected=dense[tuple(query)+(slice(None),)]
            np.testing.assert_array_equal(table[prefix+(slice(None),)],expected)
            assert table[prefix+(0,)]==expected[0]
            assert table[prefix+(1,)]==expected[1]
        del dense
    assert not table.keys.flags.writeable
    assert not table.values.flags.writeable
