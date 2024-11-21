import pytest

import numpy as np
import pandas as pd
from scipy import sparse

from analysis import sparse_pivot
from kmer import KmerDistanceCalculator


def test_testing():
    assert (1 + 1) == 2


def test_pivot():
    communities = []
    df = pd.DataFrame({"toy": ["doll", "truck", "block"], "observed": [1, 2, 20]})
    communities.append(("jack", df))
    df = pd.DataFrame(
        {"toy": ["pony", "doll", "princess dress", "block"], "observed": [1, 1, 2, 100]}
    )
    communities.append(("jill", df))
    df = pd.DataFrame(
        {"toy": ["puzzle", "truck", "matchbox car"], "observed": [2, 1, 6]}
    )
    communities.append(("john", df))

    for name, df in communities:
        df["owner"] = name
    metacomm = pd.concat([df for (name, df) in communities])

    pt1 = sparse_pivot(communities, metacomm, ["toy"])
    pt2 = metacomm.pivot_table(
        values="observed", index=["toy"], columns=["owner"], aggfunc="sum", fill_value=0
    )
    for df in [pt1, pt2]:
        df.reset_index(drop=False, inplace=True)
    for col in pt2.columns:
        assert (pt1[col] == pt2[col]).all()


def test_kmer1():
    calc = KmerDistanceCalculator(3, alphabet="ATCG")
    sequence = "AAATTTTCAT"
    counts = calc.count_kmers(sequence)
    assert counts["AAA"] == 1
    assert counts["AAT"] == 1
    assert counts["ATT"] == 1
    assert counts["TTT"] == 2
    assert counts["TTC"] == 1
    assert counts["TCA"] == 1
    assert counts["CAT"] == 1
    assert counts["GGG"] == 0
    assert counts["GCC"] == 0

    assert len(calc.omega) == 4 * 4 * 4
    assert len(set(calc.omega)) == len(calc.omega)


def test_kmer2():
    calc = KmerDistanceCalculator(3, alphabet="ATCG")
    sequence = "ACATCAGTACGTACTTACGTACGATC"
    counts = calc.count_kmers(sequence)
    vector = calc.kmer_vector(sequence, normalize=False)
    for i, kmer in enumerate(calc.omega):
        assert counts[kmer] == vector[i]


def test_kmer_norm():
    calc = KmerDistanceCalculator(3, alphabet="ATCG")
    sequence = "ACATCAGTACGTACTTCCCCCCCGGGGGGACGTACGATC"
    vector = calc.kmer_vector(sequence)
    assert np.isclose(sum([x * x for x in vector]), 1.0)


def test_kmer_sparse():
    calc = KmerDistanceCalculator(3, alphabet="ATCG")
    sequence = "ACATCAAAAAAATTTATTCATTACATTAGTACGTACTTCCCCCCCGGGGGGACGTACGATC"
    for normalize in [False, True]:
        vector_dense = np.array(calc.kmer_vector(sequence, normalize=normalize))
        V, J = calc.kmer_vector_sparse(sequence, normalize=normalize)
        assert V.shape == J.shape
        for j in J:
            assert j >= 0
            assert j < len(calc.omega)
        I = np.zeros_like(J)
        vector_sparse = sparse.coo_array(
            (V, (I, J)), shape=(1, len(calc.omega))
        ).toarray()
        assert (vector_dense == vector_sparse).all()


def test_cosine_sim():
    sequences = ["AAACC", "TTGTTT", "AAATTT"]

    calc = KmerDistanceCalculator(3, alphabet="ATCG")
    vectors = [np.array(calc.kmer_vector(seq, normalize=True)) for seq in sequences]
    assert np.isclose(np.dot(vectors[0], vectors[0]), 1.0)
    assert np.isclose(np.dot(vectors[0], vectors[1]), 0.0)
    assert 0.1 < np.dot(vectors[2], vectors[1]) < 0.9
