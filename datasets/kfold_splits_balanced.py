import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_balanced_kfold_splits(
    data_path='',
    num_splits=5,
    random_state=42,
    filename='filename.csv',
    output_filename='kfold_splits.csv',
    counts_filename='kfold_counts.csv',
    samples_per_class=12   # Number of samples (slides) you want from each class
):
    # 1. Load your dataset
    data = pd.read_csv(os.path.join(data_path, filename))

    # 2. Ensure the label column is present
    if 'label' not in data.columns:
        raise ValueError("Label column ('label') missing from the dataset.")

    # 3. Split into negative/positive
    neg_data = data[data['label'] == 'chipnegative']
    pos_data = data[data['label'] == 'chippositive']

    # --- Optionally check you have enough samples in each class ---
    if len(neg_data) < samples_per_class:
        raise ValueError(f"You requested {samples_per_class} negatives, but only {len(neg_data)} available.")
    if len(pos_data) < samples_per_class:
        raise ValueError(f"You requested {samples_per_class} positives, but only {len(pos_data)} available.")
    
    # 4. Randomly sample the desired number from each class to get a balanced set
    neg_data_sampled = neg_data.sample(n=samples_per_class, random_state=random_state)
    pos_data_sampled = pos_data.sample(n=samples_per_class, random_state=random_state)
    
    balanced_data = pd.concat([neg_data_sampled, pos_data_sampled], ignore_index=True)

    # 5. Create directory for saving (if needed)
    save_path = data_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 6. Assign a temporary fold_id using StratifiedKFold
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    balanced_data['fold_id'] = -1

    # Because we used only two classes (each with same count), 
    # StratifiedKFold will preserve the ratio 1:1 in each fold.
    for fold_number, (train_index, test_index) in enumerate(skf.split(balanced_data, balanced_data['label'])):
        balanced_data.loc[test_index, 'fold_id'] = fold_number
    
    # 7. For each of the k folds, create a new column marking each row as train/val/test
    #    using the custom pattern you had in your code comment
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        balanced_data[fold_col] = 'train'  # default

        # test_fold = (i + num_splits - 1) % num_splits
        # val_fold  = (i + num_splits - 2) % num_splits
        test_fold = (i + num_splits - 1) % num_splits
        val_fold  = (i + num_splits - 2) % num_splits

        balanced_data.loc[balanced_data['fold_id'] == val_fold,  fold_col] = 'val'
        balanced_data.loc[balanced_data['fold_id'] == test_fold, fold_col] = 'test'

    # 8. Create and save the counts for each fold
    fold_results = []
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        train_count = (balanced_data[fold_col] == 'train').sum()
        val_count   = (balanced_data[fold_col] == 'val').sum()
        test_count  = (balanced_data[fold_col] == 'test').sum()

        fold_results.append({
            'fold': i,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count
        })

    counts_df = pd.DataFrame(fold_results)
    counts_df.to_csv(os.path.join(save_path, counts_filename), index=False)

    # 9. (Optional) Remove the 'fold_id' column if not needed
    balanced_data.drop(columns=['fold_id'], inplace=True)

    # 10. Save the final CSV with k-fold splits
    balanced_data.to_csv(os.path.join(save_path, output_filename), index=False)

    print(f"\n{num_splits}-fold pattern splits (Train/Val/Test) saved to '{output_filename}'")
    print(f"K-Fold counts saved to '{counts_filename}'")

# Example usage:
in_file = "features_tiles_uni.csv"
out_file = "features_tiles_uni_kfold.csv"
out_counts = "features_tiles_uni_kfold_counts.csv"

data_path = '/ictstr01/home/aih/laia.mana/project/DATA/feature_extraction/'
generate_balanced_kfold_splits(
    data_path=data_path,
    num_splits=5,
    random_state=42,
    filename=in_file,
    output_filename=out_file,
    counts_filename=out_counts,
    samples_per_class=12
)
