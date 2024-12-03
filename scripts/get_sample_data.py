import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

reldir = Path("..")
datadir = reldir / "data"


def get_manifest_files():
    manifest_dir = datadir / "manifests"
    name_pattern = re.compile(r"MDS_Batch(\d+)_(\d+).xlsx")
    for child in manifest_dir.iterdir():
        m = name_pattern.match(child.name)
        if m:
            batch_num = int(m.group(1))
            yield child, batch_num


def get_manifests(already_sequenced_only=True):
    for one_path, batch_num in get_manifest_files():
        if already_sequenced_only:
            if batch_num > get_max_batch_sequenced():
                continue
        df = pd.read_excel(one_path, header=1)
        df["BatchNumber"] = batch_num
        yield df


def get_manifest_records():
    df = pd.concat(get_manifests())
    df["Sample ID"] = (
        "MDS_" + df["EPatID"].astype(str) + "_" + df["ECaseID"].astype(str).str.zfill(2)
    )
    df.reset_index(inplace=True)
    return df


def get_samples(cohort, set_size=100):
    """
    Deprecated

    This is (probably) not the function you want.
    Just to test getting something going--
    This chooses `set_size` samples in a given cohort, and
    also `set_size` RANDOM samples not in the given cohort.
    They may have other cancers/infections, and are not matched
    wrt age/sex/etc.

    See :func:`get_cohort_samples`

    Arguments:

    cohort-- must be one of the cohort (not category) labels in the OneHotEncoding spreadsheet

    set_size-- number of files to return (of which 50% will be in the cohort, 50% not). Not enforced
    that this is an even number. If set_size is None, the number of files returned will be 2x the
    number of patients in the given cohort.

      Note that this sorts samples by B2B ID and keeps only the first for each patient.
    Sorting by B2B ID is equivalent to sorting by time.
    Thus,
    1) will choose the EARLIEST sample for each patient
    2) the non-cohort samples selected are a fairly arbitrary set, distinguished ONLY by
    having the lowest B2B ID.
    This is not at all what we want long-term, so this will be replaced by something more
    sophisticated.
    """
    the_path = datadir / "CohortOneHotEncoding.xlsx"
    df = pd.read_excel(the_path)
    df.sort_values(by="ID", inplace=True)
    manifest_records = get_manifest_records()
    manifest_records.drop_duplicates(subset="Sample Barcode", inplace=True)
    m = df.merge(
        manifest_records,
        how="left",
        left_on="ID",
        right_on="Sample Barcode",
        validate="1:1",
        indicator=True,
    )
    df["Roche Sample ID"] = m["Sample ID"]
    df["unique pat id"] = m["EPatID"]
    df.dropna(subset="Roche Sample ID", inplace=True)
    df.drop_duplicates(
        subset="unique pat id", keep="first", inplace=True, ignore_index=True
    )
    col = "in %s cohort" % cohort
    mask = df[col]
    if set_size is None:
        class_size = mask.sum()
    else:
        class_size = set_size // 2
        if mask.sum() < class_size:
            print("Number of samples in cohort already sequenced: %d" % mask.sum())
            raise Exception("Not enough samples in cohort!")
    if (~mask).sum() < class_size:
        raise Exception("Not enough non-cohort samples!")
    mask = df[col]
    in_cohort_set = df[mask].reset_index(drop=True)
    not_in_cohort_set = df[~mask].reset_index(drop=True)
    return pd.concat(
        (in_cohort_set.iloc[:class_size], not_in_cohort_set.iloc[:class_size])
    )


def annotate_metadata(df):
    summ_file = datadir / "summary.xlsx"
    meta = pd.read_excel(summ_file)
    age_mask = meta["age"] == "90+"
    meta.loc[age_mask, "age"] = 90
    meta["age"] = meta["age"].astype(int)
    m = df.merge(meta, on="ID", how="left")
    for col in ["age", "sex", "race"]:
        df[col] = m[col]


def get_cohort_samples(cohort, set_size=100):
    """
    Arguments:

    cohort-- must be one of the cohort (not category) labels in the OneHotEncoding spreadsheet

    set_size-- number of files to return (of which 50% will be in the cohort, 50% not).
    Must be an even number. If set_size is None, the number of files returned will be 2x the
    number of patients in the given cohort.
    """
    if set_size is not None:
        if set_size % 2:
            raise ValueError("Please specify an even number.")
    the_path = datadir / "CohortOneHotEncoding.xlsx"
    df = pd.read_excel(the_path)
    df.sort_values(by="ID", inplace=True)
    manifest_records = get_manifest_records()
    manifest_records.drop_duplicates(subset="Sample Barcode", inplace=True)
    m = df.merge(
        manifest_records,
        how="left",
        left_on="ID",
        right_on="Sample Barcode",
        validate="1:1",
        indicator=True,
    )
    df["Roche Sample ID"] = m["Sample ID"]
    df["unique pat id"] = m["EPatID"]
    df.dropna(subset="Roche Sample ID", inplace=True)
    df.drop_duplicates(
        subset="unique pat id", keep="first", inplace=True, ignore_index=True
    )
    annotate_metadata(df)
    in_cohort_col = "in %s cohort" % cohort
    df["in_cohort"] = df[in_cohort_col]
    mask = df[in_cohort_col]
    if set_size is None:
        class_size = mask.sum()
    else:
        class_size = set_size // 2
        if mask.sum() < class_size:
            print("Number of samples in cohort already sequenced: %d" % mask.sum())
            raise Exception("Not enough samples in cohort!")
    if (~mask).sum() < class_size:
        raise Exception("Not enough non-cohort samples!")
    mask = df[in_cohort_col]
    in_cohort_set = df[mask].reset_index(drop=True).iloc[:class_size].copy()

    df["chosen"] = False
    for row in in_cohort_set.iterrows():
        target_age = row[1].age
        target_sex = row[1].sex
        target_race = row[1].race
        df["age_diff"] = np.abs(df["age"] - target_age)
        df["sex_diff"] = df["sex"] != target_sex
        df["race_diff"] = df["race"] != target_race
        # TODO: "in culture cohort" as control for infections
        # TODO: add batch number as thing to try to match by
        df.sort_values(
            by=[
                "chosen",
                in_cohort_col,
                "in healthy cohort",
                "sex_diff",
                "age_diff",
                "race_diff",
            ],
            ascending=[True, True, False, True, True, True],
            inplace=True,
            ignore_index=True,
        )
        df.loc[1, "chosen"] = True

    not_in_cohort_set = df[df["chosen"]].copy()
    for col in not_in_cohort_set:
        if col not in in_cohort_set:
            not_in_cohort_set.drop(columns=col, inplace=True)

    assert in_cohort_set.shape == not_in_cohort_set.shape

    assert set(in_cohort_set["ID"]).isdisjoint(set(not_in_cohort_set["ID"]))

    return pd.concat((in_cohort_set, not_in_cohort_set))


def get_batch_directories():
    name_pattern = re.compile(r"(\d\d\d\d)-(\d\d)-(\d\d)_reports_mds-batch(\d+)")
    for child in datadir.iterdir():
        m = name_pattern.match(child.name)
        if m:
            batch_num = int(m.group(4))
            yield child, batch_num


def get_max_batch_sequenced():
    return max([batch_num for _, batch_num in get_batch_directories()])


def get_sample_level_directories():
    for onedir, _ in get_batch_directories():
        sample_dir = onedir / "Sample_Level_Data_Files"
        yield sample_dir


def get_replicate_level_directories():
    for onedir, _ in get_batch_directories():
        yield onedir / "Technical_Replicate_Level_Data_Files"


def get_sample_files():
    name_pattern = re.compile(r"(MDS_\d+_\d+)_cdr3_report.csv")
    for batch_dir in get_sample_level_directories():
        for child in batch_dir.iterdir():
            m = name_pattern.match(child.name)
            if m:
                yield child, m.group(1)


def get_replicate_files():
    name_pattern = re.compile(r"(MDS_\d+_\d+)_\d+_.*_cdr3_report.csv")
    for batch_dir in get_replicate_level_directories():
        for child in batch_dir.iterdir():
            m = name_pattern.match(child.name)
            if m:
                yield child, m.group(1)


def get_sequencefile_lookup():
    d = {}
    for apath, sampleID in get_sample_files():
        d[sampleID] = apath
    return d


def get_repfile_lookup():
    d = defaultdict(list)
    for apath, sampleID in get_replicate_files():
        d[sampleID].append(apath)
    return d


def get_samplefiles(cohort="PROSTATE", set_size=70):
    filepath_lookup = get_sequencefile_lookup()
    df = get_cohort_samples(cohort, set_size)
    for row in df.iterrows():
        filepath = filepath_lookup[row[1]["Roche Sample ID"]]
        sample_id = row[1]["ID"]
        in_cohort = row[1]["in_cohort"]
        yield filepath, sample_id, in_cohort

def get_samplefile_selection(cohort="PROSTATE", set_size=70):
    filepath_lookup = get_sequencefile_lookup()
    df = get_cohort_samples(cohort, set_size)
    for sample in df["Roche Sample ID"]:
        yield filepath_lookup[sample]
