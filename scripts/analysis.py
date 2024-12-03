from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import time
import psutil
from kmer import KmerDistanceCalculator
from get_sample_data import get_samplefile_selection, get_cohort_samples
from util import drop_rows_with_mask

key_names = ["Jgene", "Vgene", "cdr3_len", "cdr3_AA"]


def sample_data_files(maxcount=6):
    count = 0
    for filepath in get_samplefile_selection(set_size=maxcount):
        if not filepath.name.endswith("cdr3_report.csv"):
            continue
        yield filepath
        count += 1
        if maxcount is not None and count >= maxcount:
            break


def name_from_filepath(filepath):
    name = filepath.name
    return name[:12]


def genes_of_type(filepath, chain="IGH", functional=True, jgene=None, equal_count=None, attributes={}):
    interesting_columns = [
        "cdr3_AA",
        "Vgene",
        "Jgene",
        "chain",
        "rearrangement_type",
        "UMI_family_size",
    ]
    df = pd.read_csv(filepath, usecols=interesting_columns)
    drop_mask = df["chain"] != chain
    drop_rows_with_mask(df, drop_mask)
    if functional:
        type_tag = "functional"
    else:
        type_tag = "nonfunctional"
    drop_mask = df["rearrangement_type"] != type_tag
    drop_rows_with_mask(df, drop_mask)
    df["cdr3_len"] = df["cdr3_AA"].str.len()
    too_short_mask = df["cdr3_len"] < 3
    if too_short_mask.sum():
        drop_rows_with_mask(df, too_short_mask)
    df.sort_values(
        by="UMI_family_size", ascending=False, inplace=True, ignore_index=True
    )
    if equal_count is not None:
        df = df.iloc[:equal_count]
        equal_count = df.shape[0]
    df.drop(columns=["UMI_family_size", "chain", "rearrangement_type"], inplace=True)
    if jgene is not None:
        mask = df["Jgene"] != jgene
        drop_rows_with_mask(df, mask)
    if equal_count is None:
        df["observed"] = 1.0
    else:
        df["observed"] = 1.0 / equal_count
    df = df.groupby(key_names).sum()
    df.reset_index(drop=False, inplace=True)
    for key, value in attributes.items():
        df[key] = value
    return df


def get_offsets(communities):
    cum_total = 0
    offsets = {}
    for name, df in communities:
        offsets[name] = cum_total
        cum_total += df.shape[0]
    return offsets


def sort_and_dedup(metacommunity, index):
    metacommunity.reset_index(drop=True, inplace=True)
    metacommunity["original_index"] = metacommunity.index
    sorted_seq = metacommunity.sort_values(by=index, ignore_index=True)
    del metacommunity
    dedup_indices = np.empty((sorted_seq.shape[0]), dtype=int)
    counter = -1
    prev = None
    for i in range(sorted_seq.shape[0]):
        row = sorted_seq.iloc[i]
        value = tuple(row[col] for col in index)
        if value != prev:
            counter += 1
            prev = value
        dedup_indices[row["original_index"]] = counter
    sorted_seq = pd.DataFrame(index=sorted_seq.groupby(index).count().index)
    return (sorted_seq, dedup_indices)


def sparse_abundances(communities, seq_count, dedup_indices):
    offsets = get_offsets(communities)
    for name, df in communities:
        offset = offsets[name]
        n = df.shape[0]
        df["dedup_index"] = dedup_indices[offset : (offset + n)]

    values = np.concatenate([np.array(df["observed"]) for (name, df) in communities])
    col_indices = np.concatenate(
        [np.full((df.shape[0],), i) for i, (name, df) in enumerate(communities)]
    )
    row_indices = np.concatenate(
        [np.array(df["dedup_index"]) for (name, df) in communities]
    )
    assert values.shape == row_indices.shape
    assert row_indices.shape == col_indices.shape
    rows = seq_count
    cols = len(communities)
    abundances = sparse.coo_array(
        (values, (row_indices, col_indices)), shape=(rows, cols)
    ).tocsr()
    return abundances


def abundances_and_dedup(communities, metacommunity, index):
    (sorted_seq, dedup_indices) = sort_and_dedup(metacommunity, index)
    abundances = sparse_abundances(communities, len(sorted_seq), dedup_indices)
    return (sorted_seq, abundances)


def sparse_pivot(communities, metacommunity, index):
    """
    Equivalent to DataFrame.pivot_table on the DataFrame
    passed as metacommunity, for a limited subset
    of cases.
    communities is a sequence of tuples. Each tuple consists
    of a name and a DataFrame. metacommunity must be the
    concatination (in order) of the DataFrames in the tuples.
    - aggfunc is implied to be "sum".
    - Rather than passing the name of the values column,
    the values column MUST be named 'observed'.
    - There is no columns parameter. Rather, the names of the
    communities (the first item in each tuple in the communities
    list) are taken to be the column names in the resulting pivot.
    - fill_value is implied to be 0
    - sort is implied to be True
    - there is no option to create margins
    """
    (sorted_seq, abundances) = abundances_and_dedup(communities, metacommunity, index)
    for i, (name, df) in enumerate(communities):
        sorted_seq[name] = abundances[:, [i]].toarray()[:, 0]
    return sorted_seq


def calculate_similarity(kmers, from_start, from_end, to_start, to_end):
    a = kmers[from_start:from_end]
    b = kmers[to_start:to_end].T
    return a @ b


def make_kmer_vectors(sorted_seq):
    calc = KmerDistanceCalculator(3)
    value_arrays = []
    col_index_arrays = []
    row_index_arrays = []
    for i, row in enumerate(sorted_seq.itertuples()):
        value_array, col_index_array = calc.kmer_vector_sparse(row.cdr3_AA)
        row_index_array = np.full_like(value_array, i)
        value_arrays.append(value_array)
        col_index_arrays.append(col_index_array)
        row_index_arrays.append(row_index_array)
    value_array = np.concatenate(value_arrays)
    col_index_array = np.concatenate(col_index_arrays)
    row_index_array = np.concatenate(row_index_arrays)
    kmers = sparse.coo_array(
        (value_array, (row_index_array, col_index_array)),
        shape=(sorted_seq.shape[0], len(calc.omega)),
    ).tocsr()
    return kmers


def make_similarity(sorted_seq):
    # TODO: these should be tweakable
    max_load = 1024 * 1024 * 16
    max_strip_size = max_load * 4
    kmers = make_kmer_vectors(sorted_seq)
    n = sorted_seq.shape[0]
    # Motivation for selection of lil_array:
    # bsr_array does not support slicing, so how to assign to rectangular section?
    # coo_array or dia_array does not support item assignment, so how to assign?
    # csc_array or csr_array, when assigning, gives a warning that changing the sparsity ineffecient, lil recommended
    # dok_array is vastly slower; it's both vastly slower to assign by numeric index, and to convert to csr format.
    sorted_seq["index"] = sorted_seq.index
    breaks = sorted_seq.drop_duplicates(subset=["Jgene", "Vgene", "cdr3_len"]).drop(
        columns="cdr3_AA"
    )
    lil = None
    for i in range(breaks.shape[0]):
        for j in range(i, breaks.shape[0]):
            if breaks.iloc[i]["Vgene"] != breaks.iloc[j]["Vgene"]:
                break
            if breaks.iloc[i]["Jgene"] != breaks.iloc[j]["Jgene"]:
                break
            len_diff = breaks.iloc[j]["cdr3_len"] - breaks.iloc[i]["cdr3_len"]
            if len_diff > 4:  # TODO: parameterize this
                break
            from_start = breaks.iloc[i]["index"]
            if i + 1 < breaks.shape[0]:
                from_end = breaks.iloc[i + 1]["index"]
            else:
                from_end = n
            to_start = breaks.iloc[j]["index"]
            if j + 1 < breaks.shape[0]:
                to_end = breaks.iloc[j + 1]["index"]
            else:
                to_end = n
            chunk_size = int(max_strip_size / (to_end - to_start))
            if chunk_size < 1:
                chunk_size = 1
            for from_start_chunk in range(from_start, from_end, chunk_size):
                from_end_chunk = from_start_chunk + chunk_size
                if from_end_chunk > from_end:
                    from_end_chunk = from_end
                s = calculate_similarity(
                    kmers, from_start_chunk, from_end_chunk, to_start, to_end
                )
                if lil is None:
                    lil = sparse.lil_array((n, n), dtype=float)
                lil[from_start_chunk:from_end_chunk, to_start:to_end] = s
                if from_start != to_start:
                    lil[to_start:to_end, from_start_chunk:from_end_chunk] = s.T
                if lil.nnz >= max_load:
                    yield lil.tocsr()
                    lil = None
    if lil is not None:
        yield lil.tocsr()


def concat_community(communities):
    sequences = pd.concat([df for (name, df) in communities])
    n = sequences.shape[0]
    return sequences, n


def convert_index_to_columns(df):
    df.reset_index(drop=False, inplace=True)


def profile_similiarity(similarity):
    """
    Print out kind of a histogram to show distribution of
    non-zero similarity values.
    As it is currently written, do not call this on a large dataset.
    """
    dense = similarity.toarray()
    num_bins = 20
    count = dense.shape[0] * (dense.shape[0])
    cum = 0.0
    for x in range(num_bins):
        lower = x / num_bins
        upper = (x + 1) / num_bins
        mask = (dense > lower) & (dense <= upper)
        frac = mask.sum() / count
        cum += frac
        print("(%f, %f]: %f : %f" % (lower, upper, frac, cum))
    print(similarity.nnz / count)


def get_distinct_values(col="Jgene"):
    filename = f"{col}.txt"
    cachepath = Path("..") / "cached" / filename
    if cachepath.is_file():
        with open(cachepath, "r") as f:
            all_j = [s.strip() for s in f.readlines()]
    else:
        all_j = set()
        for filepath in sample_data_files():
            df = genes_of_type(filepath)
            all_j |= set(df[col].unique())
        all_j = sorted(list(all_j))
        with open(cachepath, "w") as f:
            for j_gene in all_j:
                f.write(f"{j_gene}\n")
    return all_j


# TODO: this seems inaccurate in its assessment of memory pressure; improve
def note_ram(quit_on_swap=False):
    mem = psutil.virtual_memory()
    if mem.percent > 75:
        print("SWAP!")
    if quit_on_swap:
        assert mem.percent <= 75
    return mem.percent


def get_metacommunity(file_count):
    equal_count = 4000
    total_n = 0
    all_effective_counts = None
    # TODO: only call sample_data_files once, not for each jgene
    for j_gene in get_distinct_values("Jgene"):
        communities = [
            (
                name_from_filepath(filepath),
                genes_of_type(filepath, jgene=j_gene, equal_count=equal_count),
            )
            for filepath in sample_data_files(file_count)
        ]
        sequences, n = concat_community(communities)
        print("Did concat_community")
        if sequences.shape[0] < 1:
            continue

        (sorted_seq, abundances) = abundances_and_dedup(
            communities, sequences, key_names
        )
        print("Did abundances_and_dedup")
        del communities
        del sequences
        convert_index_to_columns(sorted_seq)
        effective_counts = None
        for similarity in make_similarity(sorted_seq):
            if effective_counts is None:
                effective_counts = similarity @ abundances
            else:
                effective_counts = effective_counts + (similarity @ abundances)
            note_ram(False)
        if all_effective_counts is None:
            all_effective_counts = effective_counts
            all_abundances = abundances
        else:
            all_effective_counts = sparse.vstack(
                (all_effective_counts, effective_counts)
            )
            all_abundances = sparse.vstack((all_abundances, abundances))
        total_n += n
        del sorted_seq
        del effective_counts
        del similarity
        del abundances
    print(all_effective_counts.todense())
    return total_n, all_effective_counts, all_abundances


def do_diversity(filecount=8):
    total_n, all_effective_counts, all_abundances = get_metacommunity(filecount)
    all_effective_counts = all_effective_counts.todense()
    is_nonzero = all_effective_counts > 1e-9
    result = np.zeros(shape=all_effective_counts.shape, dtype=np.float64)
    np.power(all_effective_counts, -1, where=is_nonzero, out=result)
    for i in range(all_effective_counts.shape[1]):
        a = result[:, i].T
        b = all_abundances[:, [i]].todense()
        diversity = a @ b
        print(diversity)


def big_o_what():
    times = {}
    for file_count in [16, 32, 64, 75, 88, 100]:
        t0 = time.time()
        do_diversity(file_count)
        t1 = time.time()
        print("Time: ", (t1 - t0))
        times[file_count] = t1 - t0
    print()
    for n, seconds in times.items():
        print(n, seconds, n / seconds)


"""
how to view memory usage for data
print("MB:")
print(communities.memory_usage() / (1024*1024))
"""

if __name__ == "__main__":
    do_diversity(filecount=None)
    
