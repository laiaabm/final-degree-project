# k-fold cross validation (k=5) pattern:
# Fold 0: [-- | -- | -- | ** | ##]
# Fold 1: [-- | -- | ** | ## | --]
# Fold 2: [-- | ** | ## | -- | --]
# Fold 3: [** | ## | -- | -- | --]
# Fold 4: [## | -- | -- | -- | **]

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold


def load_data(file_path, label_col):
    """
    Load CSV and filter valid labels.
    """
    data = pd.read_csv(file_path)
    return data[data[label_col].isin(['chippositive', 'chipnegative'])]

def create_kfold(df, sample_col, label_col, num_splits, random_state):
    """
    Assign stratified fold IDs to each unique sample.
    """
    unique_samples = df[[sample_col, label_col]].drop_duplicates().reset_index(drop=True)
    unique_samples['fold_id'] = -1

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for fold, (_, test_idx) in enumerate(skf.split(unique_samples, unique_samples[label_col])):
        unique_samples.loc[test_idx, 'fold_id'] = fold

    return df.merge(unique_samples[[sample_col, 'fold_id']], on=sample_col, how='inner')

def assign_kfold(df, num_splits):
    """
    Create kfold columns with 'train', 'val', and 'test' roles.
    """
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        df[fold_col] = 'train'

        test_fold = (i - 1) % num_splits
        val_fold = (i - 2) % num_splits

        df.loc[df['fold_id'] == val_fold, fold_col] = 'val'
        df.loc[df['fold_id'] == test_fold, fold_col] = 'test'

    return df

def save_counts(df, save_path, counts_filename, num_splits):
    """
    Save the count of train/val/test per fold.
    """
    results = []
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        results.append({
            'fold': i,
            'train_count': (df[fold_col] == 'train').sum(),
            'val_count':   (df[fold_col] == 'val').sum(),
            'test_count':  (df[fold_col] == 'test').sum()
        })

    counts_df = pd.DataFrame(results)
    counts_df.to_csv(os.path.join(save_path, counts_filename), index=False)

def generate_sample_kfold_splits(
    data_path,
    filename,
    output_filename,
    counts_filename,
    num_splits=5,
    random_state=42,
    sample_col='sample_number',
    label_col='label'
):
    """
    Full pipeline: load, split, assign folds, and save.
    """
    file_path = os.path.join(data_path, filename)
    df = load_data(file_path, label_col)

    if sample_col not in df.columns or label_col not in df.columns:
        raise ValueError("Missing required columns in input data.")

    os.makedirs(data_path, exist_ok=True)

    df = create_kfold(df, sample_col, label_col, num_splits, random_state)
    df = assign_kfold(df, num_splits)

    save_counts(df, data_path, counts_filename, num_splits)

    df.drop(columns=['fold_id'], inplace=True)
    df.to_csv(os.path.join(data_path, output_filename), index=False)

    print(f"{num_splits}-fold patient-level splits saved to '{output_filename}'")
    print(f"K-Fold counts saved to '{counts_filename}'")


def main():
    data_path = "/ictstr01/home/aih/laia.mana/project/DATA/attention_visualization/kfold/"

    generate_sample_kfold_splits(
        data_path=data_path,
        filename="features_dino.csv",
        output_filename="kfold_dino.csv",
        counts_filename="counts_dino.csv",
        num_splits=5,
        random_state=42,
        sample_col='sample_number',
        label_col='label'
    )

if __name__ == "__main__":
    main()