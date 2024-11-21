import pandas as pd
import numpy as np
from get_sample_data import get_manifest_records, get_repfile_lookup
from util import drop_rows_with_mask

filepath_lookup = get_repfile_lookup()


def samples_on_plate(platenum):
    df = get_manifest_records()

    mask = df["iPETE_Temp_Plate_Number"] != platenum
    drop_rows_with_mask(df, mask)
    drop_rows_with_mask(df, df["Sample Type"] != "Peripheral Blood Mononuclear Cells")
    df.drop_duplicates(subset="Sample ID", inplace=True)
    for sample in df["Sample ID"]:
        for filepath in filepath_lookup[sample]:
            yield filepath


def cell_counts_on_plate(platenum):
    for filepath in samples_on_plate(platenum):
        df = pd.read_csv(filepath)
        drop_rows_with_mask(df, df["rearrangement_type"] != "functional")
        drop_rows_with_mask(df, df["chain"] == "chimera")
        yield df.shape[0]


def good_reads_on_plate(platenum):
    for filepath in samples_on_plate(platenum):
        df = pd.read_csv(filepath)
        reads = df["UMI_family_size"].sum()
        yield reads


for platenum in range(4, 45):
    print("Plate: %d" % platenum)
    counts = [n for n in good_reads_on_plate(platenum)]
    print("Median reads: %d x 10e6" % round(np.median(counts) / 1000000))
"""    
for platenum in [13, 26]:
    for filepath in samples_on_plate(platenum):
        print(filepath)
"""
