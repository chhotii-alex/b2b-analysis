import pytest

from analysis import sparse_pivot
import pandas as pd

def test_testing():
    assert (1+1) == 2

def test_pivot():
    communities = []
    df = pd.DataFrame({
        "toy": ["doll", "truck", "block"],
        "observed": [1, 2, 20]})
    communities.append( ('jack', df) )
    df = pd.DataFrame({
        "toy": ["pony", "doll", "princess dress", "block"],
        "observed": [1, 1, 2, 100]})
    communities.append( ('jill', df) )
    df = pd.DataFrame({
        "toy": ["puzzle", "truck", "matchbox car"],
        "observed": [2, 1, 6]})
    communities.append( ('john', df) )
    
    for name, df in communities:
        df['owner'] = name
    metacomm = pd.concat([df for (name, df) in communities])

    pt1 = sparse_pivot(communities, metacomm, ['toy'])
    pt2 = metacomm.pivot_table(values="observed",
                               index=['toy'],
                               columns=['owner'],
                               aggfunc="sum", fill_value=0)
    for df in [pt1, pt2]:
        df.reset_index(drop=False, inplace=True)
    for col in pt2.columns:
        assert (pt1[col] == pt2[col]).all()

