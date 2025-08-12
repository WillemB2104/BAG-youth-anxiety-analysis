import datetime
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from tqdm.notebook import tqdm

from .evaluation_classifier import Evaluater


def ensure_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def string_to_dict(string, pattern):
    regex = re.sub(r'{(.+?)}', r'(?P<_\1>.+)', pattern)
    values = list(re.search(regex, string).groups())
    keys = re.findall(r'{(.+?)}', pattern)
    _dict = dict(zip(keys, values))
    return _dict


def reorder_df_col(df, column_to_insert, column_before_insert):
    df.insert(df.columns.get_loc(column_before_insert), column_to_insert, df.pop(column_to_insert))


def load_most_recent_WG_data(WG):
    if WG == 'PD':
        working_group_dir = '/data/wbbruin/Documents/Stage-Machine-Learning/ENIGMA_PANIC/POOLED/'
    elif WG == 'SAD':
        working_group_dir = '/data/wbbruin/Documents/Stage-Machine-Learning/ENIGMA_SAD/'
    elif WG == 'GAD':
        working_group_dir = '/data/wbbruin/Documents/Stage-Machine-Learning/ENIGMA_GAD/'

    files = glob(os.path.join(working_group_dir, '*POOLED_DATA_FOR_CROSS_DISORDER.csv'))

    if len(files) > 0:
        print("Loading most recent pooled data...")
        most_recent_idx = np.argmax([datetime.datetime.strptime(
            f.split('_POOLED_DATA_FOR_CROSS_DISORDER')[0].split('_')[-1], "%Y-%m-%d") for f in files])
        print("Found data stored @ {}".format(files[most_recent_idx]))
        df = pd.read_csv(files[most_recent_idx], low_memory=False)

        return df
    else:
        print("No data found!")


def merge_bool_masks(*masks):
    return np.array([all(tup) for tup in zip(*masks)])


def has_N_per_class(data_df, class_label, subject_mask=None, N_threshold_c0=10, N_threshold_c1=1, verbose=True):
    groups = np.array(data_df.MultiSiteID.values)

    if subject_mask is not None:
        tmp_df = data_df.loc[subject_mask, ['SubjID', class_label, 'MultiSiteID']].copy()
    else:
        tmp_df = data_df.loc[:, ['SubjID', class_label, 'MultiSiteID']].copy()

    all_sites = np.unique(tmp_df.MultiSiteID)
    unique_class_values = sorted(np.unique(tmp_df[class_label]))
    assert len(unique_class_values) == 2

    # Store number of included subjects per class and site in Dataframe
    counts_df = tmp_df.groupby(['MultiSiteID', class_label]).size()
    counts_df = pd.DataFrame(counts_df)
    counts_df = counts_df.reset_index()
    counts_df = counts_df.pivot(index='MultiSiteID', columns=class_label)
    counts_df.columns = counts_df.columns.droplevel(0)
    counts_df.columns.name = None
    counts_df = counts_df.reset_index()

    if N_threshold_c0 + N_threshold_c1 > 0:
        included_sites = counts_df.loc[(counts_df[unique_class_values[0]] >= N_threshold_c0) &
                                       (counts_df[unique_class_values[1]] >= N_threshold_c1)]['MultiSiteID'].values
        excluded_sites = list(set(all_sites).difference(included_sites))
    else:
        included_sites = all_sites
        excluded_sites = []

    counts_df = counts_df.loc[counts_df.MultiSiteID.isin(included_sites)]
    site_mask = [g not in excluded_sites for g in groups]

    if verbose:
        N_excluded = len(tmp_df.loc[tmp_df.MultiSiteID.isin(excluded_sites)])
        print(f"Excluded {N_excluded} subjects belonging to {len(excluded_sites)} different sites:")
        excluded_sites_str = '\n'.join(excluded_sites)
        print(f"{excluded_sites_str}")
    return site_mask, counts_df


def exclude_subjects_with_missing_features(df, FS_cols, completeness_threshold=0.75):
    # Extract FS features
    X = df[FS_cols].values

    N_features = len(FS_cols)

    # Create mask for subjects that have too many missing values
    N_missing_per_subject = np.sum(np.isnan(X), axis=1)
    p_missing_per_subject = N_missing_per_subject / float(N_features)
    p_missing_inclusion_mask = (p_missing_per_subject < (1 - completeness_threshold))
    n_missing_excluded = sum(~p_missing_inclusion_mask)

    print(f"{sum(N_missing_per_subject > 0)} of {len(N_missing_per_subject)} subjects have >=1 missing features")
    print(f"{n_missing_excluded} subjects excluded with >{int((1 - completeness_threshold) * 100)}% missing features")
    print(df.loc[~p_missing_inclusion_mask].groupby(['WG', 'Dx']).size())
    print()

    df = df.loc[p_missing_inclusion_mask]

    return df


def extract_FS_cols(SPREADSHEET_TEMPLATES_DIR):
    spreadsheet_files = os.listdir(SPREADSHEET_TEMPLATES_DIR)
    spreadsheet_columns = {}

    for f in spreadsheet_files:

        if '.csv' in f:
            df = pd.read_csv(os.path.join(SPREADSHEET_TEMPLATES_DIR, f))
        else:
            df = pd.read_excel(os.path.join(SPREADSHEET_TEMPLATES_DIR, f), header=1)

        spreadsheet_columns[f] = df.columns.values

    # Remove columns that are duplicated across sheets
    for f in spreadsheet_files:

        if f != 'CorticalMeasuresENIGMA_SurfAvg.csv':

            if ('LSurfArea' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'LSurfArea')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

            if ('RSurfArea' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'RSurfArea')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

        if f != 'CorticalMeasuresENIGMA_ThickAvg.csv':

            if ('LThickness' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'LThickness')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

            if ('RThickness' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'RThickness')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

        if f != 'LandRvolumes.csv':

            if ('ICV' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'ICV')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

    # Extract FreeSurfer labels
    FS_cols = np.concatenate([
        spreadsheet_columns['CorticalMeasuresENIGMA_SurfAvg.csv'][1:],
        spreadsheet_columns['CorticalMeasuresENIGMA_ThickAvg.csv'][1:],
        spreadsheet_columns['LandRvolumes.csv'][1:]
    ])

    print("Total FS columns: {}".format(len(FS_cols)))

    # Create a subset without global features (i.e. summarized measures over hemipsheres and ICV)
    global_FS_features = ['LSurfArea', 'RSurfArea', 'LThickness', 'RThickness', 'ICV']
    subset_mask = [f not in global_FS_features for f in FS_cols]
    FS_cols_wo_global = FS_cols[subset_mask]

    print("Total FS columns without global hemishpere measures and ICV: {}".format(len(FS_cols_wo_global)))

    # Parse out different modalities (CT/CSA/SUBVOL)
    ct_mask = ['thick' in f for f in FS_cols_wo_global]
    csa_mask = ['surf' in f for f in FS_cols_wo_global]
    subcort_mask = ~np.array(ct_mask) & ~np.array(csa_mask)

    assert sum(ct_mask) + sum(csa_mask) + sum(subcort_mask) == len(FS_cols_wo_global)

    return FS_cols, FS_cols_wo_global