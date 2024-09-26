from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import time

key_names = ["Jgene", "Vgene", "cdr3_len", "cdr3_AA"]

reldir = Path("..")
datadir = reldir / "data"
samplesdir = datadir / "Sample_Level_Data_Files"

def sample_data_files():
    for filepath in samplesdir.iterdir():
        if not filepath.name.endswith("cdr3_report.csv"):
            continue
        yield filepath

def drop_rows_with_mask(df, drop_mask):
    dropped_indices = df[drop_mask].index
    df.drop(index=dropped_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

def name_from_filepath(filepath):
    name = filepath.name
    return name[:12]
    
def genes_of_type(filepath,
                  chain="IGH",
                  functional=True):
    interesting_columns = [
        'cdr3_AA',
        'Vgene',
        'Jgene',
        'chain',
        'rearrangement_type',
        'biosample_name']
    df = pd.read_csv(filepath, usecols=interesting_columns)
    drop_mask = df['chain'] != chain
    drop_rows_with_mask(df, drop_mask)
    if functional:
        type_tag = "functional"
    else:
        type_tag = "nonfunctional"
    drop_mask = df['rearrangement_type'] != type_tag
    drop_rows_with_mask(df, drop_mask)
    df.drop(columns=['chain', 'rearrangement_type'], inplace=True)
    df['cdr3_len'] = df["cdr3_AA"].str.len()
    too_short_mask = (df['cdr3_len'] < 3)
    if too_short_mask.sum():
        drop_rows_with_mask(df, too_short_mask)
    df["observed"] = 1
    df = df.groupby(key_names + ['biosample_name']).count()
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
        dedup_indices[row['original_index']] = counter
    sorted_seq = pd.DataFrame(index=sorted_seq.groupby(index).count().index)
    return (sorted_seq, dedup_indices)

def sparse_abundances(communities, seq_count, dedup_indices): 
    offsets = get_offsets(communities)
    for name, df in communities:
        offset = offsets[name]
        n = df.shape[0]
        df["dedup_index"] = dedup_indices[offset:(offset+n)]

    V = np.concatenate([np.array(df['observed']) for (name, df) in communities])
    J = np.concatenate([np.full((df.shape[0],), i) for i, (name, df) in enumerate(communities)])
    I = np.concatenate([np.array(df['dedup_index']) for (name, df) in communities])
    assert V.shape == I.shape
    assert I.shape == J.shape
    rows = seq_count
    cols = len(communities)
    abundances = sparse.coo_array((V,(I,J)),shape=(rows, cols)).tocsr()
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

def get_metacommunity(use_sparse=True):
    communities = [(name_from_filepath(filepath), genes_of_type(filepath))
                for filepath in sample_data_files()]
    sequences = pd.concat([df for (name, df) in communities])
    for name, df in communities:
        df.drop(columns='biosample_name', inplace=True)

    if use_sparse:
        return sparse_pivot(communities, sequences,
                            index=key_names)
    else:
        return pd.pivot_table(sequences,
                                 values="observed",
                                 index=key_names,
                                 columns=['biosample_name'],
                                 aggfunc="sum", fill_value=0)

# To time something put it between these lines:
#t0 = time.time()
#t1 = time.time()
#print("Time: ", (t1-t0))

def make_metacommunities():
    c = {}
    for option in [False, True]:
        metacommunity = get_metacommunity(option)
        metacommunity.reset_index(inplace=True)
        c[option] = metacommunity

    # test that we got the same results either way
    for col in c[True].columns:
        assert (c[True][col] == c[False][col]).all()

    print(c[True].head(30))
    print(c[True].shape)


make_metacommunities()
    
"""
how to view memory usage for data
print("MB:")
print(communities.memory_usage() / (1024*1024))
"""
        
