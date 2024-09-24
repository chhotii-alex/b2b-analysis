from pathlib import Path
import numpy as np
import pandas as pd
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

def sparse_pivot(communities, sequences):
    sequences.reset_index(drop=True, inplace=True)
    offsets = get_offsets(communities)
    sequences["original_index"] = sequences.index
    sorted_seq = sequences.sort_values(by=key_names, ignore_index=True)
    sequences['dedup_index'] = 0
    dedup_indices = np.empty((sequences.shape[0]), dtype=int)
    counter = -1
    prev = None
    for i in range(sorted_seq.shape[0]):
        row = sorted_seq.iloc[i]
        value = tuple(row[col] for col in key_names)
        if value != prev:
            counter += 1
            prev = value
        dedup_indices[row['original_index']] = counter
    sorted_seq = pd.DataFrame(index=sorted_seq.groupby(key_names).count().index)

    for name, df in communities:
        offset = offsets[name]
        n = df.shape[0]
        df["dedup_index"] = dedup_indices[offset:(offset+n)]

    for i, (name, df) in enumerate(communities):
        abundances = np.zeros((sorted_seq.shape[0]), dtype=int)
        for row in df.itertuples():
            dedup_index = row.dedup_index
            abundances[dedup_index] += row.observed
        sorted_seq[name] = abundances
    return sorted_seq

def get_communities(use_sparse=True):
    communities = [(name_from_filepath(filepath), genes_of_type(filepath))
                for filepath in sample_data_files()]
    sequences = pd.concat([df for (name, df) in communities])

    if use_sparse:
        return sparse_pivot(communities, sequences)
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

c = {}
for option in [True, False]:
    communities = get_communities(option)
    communities.reset_index(inplace=True)
    c[option] = communities

for col in c[True].columns:
    if not (c[True][col] == c[False][col]).all():
        print("Discrepancy!", col)

print("MB:")
print(communities.memory_usage() / (1024*1024))
print(communities.head(30))
print(communities.shape)
        
