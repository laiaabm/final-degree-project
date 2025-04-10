# k-fold cross validation (k=5) pattern:
# Fold 0: [-- | -- | -- | ** | ##]
# Fold 1: [-- | -- | ** | ## | --]
# Fold 2: [-- | ** | ## | -- | --]
# Fold 3: [** | ## | -- | -- | --]
# Fold 4: [## | -- | -- | -- | **]

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

def generate_sample_kfold_splits(
    data_path='',
    num_splits=5,
    random_state=42,
    filename='filename.csv',
    output_filename='kfold_splits.csv',
    counts_filename='kfold_counts.csv',
    label_col='label',
    sample_col='sample_number'
):
    # 1. Load dataset
    data = pd.read_csv(os.path.join(data_path, filename))

    # 2. Filter out any rows with 'chipunknown' label
    data = data[data[label_col].isin(['chippositive', 'chipnegative'])]

    # 3. Check required columns
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    if sample_col not in data.columns:
        raise ValueError(f"Sample column '{sample_col}' not found.")

    # 4. Create save path if not exists
    save_path = data_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 5. Create unique samples df
    sample_df = data[[sample_col, label_col]].drop_duplicates().reset_index(drop=True)

    # Initialize a column in 'sample_df' to store fold labels (0 to 4)
    sample_df['fold_id'] = -1

    # 6. Assign folds using StratifiedKFold on unique samples
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for fold_number, (train_index, test_index) in enumerate(skf.split(sample_df, sample_df[label_col])):
        sample_df.loc[test_index, 'fold_id'] = fold_number

    # 7. Merge back fold info into original data
    data = data.merge(sample_df[[sample_col, 'fold_id']], on=sample_col, how='inner')

    # 8. Create fold split columns for each k-fold
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        data[fold_col] = 'train'  # default

        test_fold = (i + num_splits - 1) % num_splits  # one step behind i
        val_fold  = (i + num_splits - 2) % num_splits  # two steps behind i

        data.loc[data['fold_id'] == val_fold,  fold_col] = 'val'
        data.loc[data['fold_id'] == test_fold, fold_col] = 'test'

    # 9. Save counts per fold
    fold_results = []
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        train_count = (data[fold_col] == 'train').sum()
        val_count   = (data[fold_col] == 'val').sum()
        test_count  = (data[fold_col] == 'test').sum()

        fold_results.append({
            'fold': i,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count
        })

    counts_df = pd.DataFrame(fold_results)
    counts_df.to_csv(os.path.join(save_path, counts_filename), index=False)

    # 10. Drop helper fold column and save final splits
    data.drop(columns=['fold_id'], inplace=True)
    data.to_csv(os.path.join(save_path, output_filename), index=False)

    print(f"\n{num_splits}-fold patient-level splits (Train/Val/Test) saved to '{output_filename}'")
    print(f"K-Fold counts saved to '{counts_filename}'")

# Example usage:
in_file = "features_224_dinobloom-g.csv"
out_file = "kfold_224_dinobloom-g.csv"
out_counts = "counts_224_dinobloom-g.csv"

data_path = '/ictstr01/home/aih/laia.mana/project/DATA/kfold/'

generate_sample_kfold_splits(
    data_path=data_path,
    num_splits=5,
    random_state=42,
    filename=in_file,
    output_filename=out_file,
    counts_filename=out_counts,
    sample_col='sample_number',
    label_col='label'
)
