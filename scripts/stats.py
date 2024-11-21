from get_sample_data import get_sample_files
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import describe


def all_actual_sample_data():
    for file_path, sample_id in get_sample_files():
        if "99999" in sample_id:
            continue
        yield file_path


def get_sample_data_n():
    count = 0
    for filepath in all_actual_sample_data():
        count += 1
    return count


def get_gene_counts():
    results = defaultdict(list)
    progress = tqdm(total=get_sample_data_n())
    for filepath in all_actual_sample_data():
        interesting_columns = ["chain", "rearrangement_type"]
        df = pd.read_csv(filepath, usecols=interesting_columns)
        for chain_type in ["IGH", "TRB", "TRD"]:
            for rearrangement_type in ["functional", "nonfunctional"]:
                mask = (df["chain"] == chain_type) & (
                    df["rearrangement_type"] == rearrangement_type
                )
                results[(rearrangement_type, chain_type)].append(mask.sum())
        progress.update(1)
    progress.close()
    return results


"""
for chain_type in ['IGH', 'TRB', 'TRD']:
    for rearrangement_type in ["functional", "nonfunctional"]:
        print(chain_type, rearrangement_type)
        values = results[(rearrangement_type, chain_type)]
        print(describe(values))
        print(np.histogram(values))
        print()
"""
