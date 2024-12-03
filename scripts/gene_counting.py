import pandas as pd
from get_sample_data import get_samplefiles, get_cohort_samples
from analysis import genes_of_type
from pymmunomics.preprocessing.repertoire import count_features

def count_genes():
    df = pd.concat([genes_of_type(f, attributes={"Sample": b2bid,
                                                 "disease": group})
                    for f, b2bid, group in get_samplefiles("LUNG", 70)])
    print(df.columns)
    results = count_features(repertoire=df,
                   repertoire_groups=["Sample", "disease"],
                   clonotype_features=["Vgene", "Jgene"])
    return results


if __name__ == "__main__":
    count_genes()
