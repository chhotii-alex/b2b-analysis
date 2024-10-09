from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import time
import psutil
from kmer import KmerDistanceCalculator

key_names = ["Jgene", "Vgene", "cdr3_len", "cdr3_AA"]

reldir = Path("..")
datadir = reldir / "data"
samplesdir = datadir / "Sample_Level_Data_Files"


def sample_data_files(maxcount=6):
    count = 0
    for filepath in samplesdir.iterdir():
        if not filepath.name.endswith("cdr3_report.csv"):
            continue
        yield filepath
        count += 1
        if count >= maxcount:
            break

def drop_rows_with_mask(df, drop_mask):
    dropped_indices = df[drop_mask].index
    df.drop(index=dropped_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)


def name_from_filepath(filepath):
    name = filepath.name
    return name[:12]


def genes_of_type(filepath, chain="IGH", functional=True,
                  jgene=None):
    interesting_columns = [
        "cdr3_AA",
        "Vgene",
        "Jgene",
        "chain",
        "rearrangement_type",
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
    df.drop(columns=["chain", "rearrangement_type"], inplace=True)
    if jgene is not None:
        mask = df['Jgene'] != jgene
        drop_rows_with_mask(df, mask)
    df["cdr3_len"] = df["cdr3_AA"].str.len()
    too_short_mask = df["cdr3_len"] < 3
    if too_short_mask.sum():
        drop_rows_with_mask(df, too_short_mask)
    df["observed"] = 1
    df = df.groupby(key_names).count()
    df.reset_index(drop=False, inplace=True)
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

    V = np.concatenate([np.array(df["observed"]) for (name, df) in communities])
    J = np.concatenate(
        [np.full((df.shape[0],), i) for i, (name, df) in enumerate(communities)]
    )
    I = np.concatenate([np.array(df["dedup_index"]) for (name, df) in communities])
    assert V.shape == I.shape
    assert I.shape == J.shape
    rows = seq_count
    cols = len(communities)
    abundances = sparse.coo_array((V, (I, J)), shape=(rows, cols)).tocsr()
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
    Vs = []
    Js = []
    Is = []
    for i, row in enumerate(sorted_seq.itertuples()):
        V, J = calc.kmer_vector_sparse(row.cdr3_AA)
        I = np.full_like(V, i)
        Vs.append(V)
        Js.append(J)
        Is.append(I)
    V = np.concatenate(Vs)
    J = np.concatenate(Js)
    I = np.concatenate(Is)
    kmers = sparse.coo_array(
        (V, (I, J)), shape=(sorted_seq.shape[0], len(calc.omega))
    ).tocsr()
    return kmers


def make_similarity(sorted_seq):
    # TODO: these should be tweakable
    max_load = 1024*1024*64
    max_strip_size = max_load*4
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
            chunk_size = int(max_strip_size/(to_end - to_start))
            if chunk_size < 1:
                chunk_size = 1
            for from_start_chunk in range(from_start, from_end, chunk_size):
                from_end_chunk = from_start_chunk + chunk_size
                if from_end_chunk > from_end:
                    from_end_chunk = from_end
                s = calculate_similarity(kmers, from_start_chunk, from_end_chunk, to_start, to_end)
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
    count = dense.shape[0]*(dense.shape[0])
    cum = 0.0
    for x in range(num_bins):
        lower = x/num_bins
        upper = (x+1)/num_bins
        mask = (dense > lower) & (dense <= upper)
        frac = mask.sum()/count
        cum += frac
        print("(%f, %f]: %f : %f" % (lower, upper, frac, cum))
    print(similarity.nnz / count)

def get_distinct_values(file_count, col='Jgene'):
    all_j = set()
    for filepath in sample_data_files(file_count):
        df = genes_of_type(filepath)
        all_j |= set(df[col].unique())
    return sorted(list(all_j))

def note_ram(quit_on_swap=False):
    mem = psutil.virtual_memory()
    if mem.percent > 75:
        print("SWAP!")
    if quit_on_swap:
        assert mem.percent <= 75
    return mem.percent

def get_metacommunity(file_count):
    """
    Things to try:
    * try all different sparse data structure options
    * do similarity in stripes (like greylock (use greylock?))
    """
    total_n = 0
    all_effective_counts = None
    for j_gene in  get_distinct_values(file_count, 'Jgene'):
        communities = [
            (name_from_filepath(filepath), genes_of_type(filepath, jgene=j_gene))
            for filepath in sample_data_files(file_count)
        ]
        sequences, n = concat_community(communities)
        print("Did concat_community")
        if sequences.shape[0] < 1:
            continue

        (sorted_seq, abundances) = abundances_and_dedup(communities, sequences, key_names)
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
            note_ram(True)
        if all_effective_counts is None:
            all_effective_counts = effective_counts
        else:
            all_effective_counts = sparse.vstack((all_effective_counts, effective_counts))
        total_n += n
        del sorted_seq
        del effective_counts
        del similarity
        del abundances
    return total_n, all_effective_counts

def big_o_what():
    times = {}
    for file_count in [2, 4, 8, 16]:
        t0 = time.time()
        n, _ = get_metacommunity(file_count)
        t1 = time.time()
        print("Time: ", (t1-t0))
        print(n / (t1-t0))
        times[n] = (t1-t0)
    print()
    for n, seconds in times.items():
        print(n, seconds, n/seconds)
        
"""
how to view memory usage for data
print("MB:")
print(communities.memory_usage() / (1024*1024))
"""

if __name__ == '__main__':
    get_metacommunity(8)
