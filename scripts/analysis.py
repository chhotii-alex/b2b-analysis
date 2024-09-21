from pathlib import Path
import pandas as pd

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
    return df

sequences = pd.concat([
    genes_of_type(filepath) for filepath in sample_data_files()])
print(sequences)

    
        
