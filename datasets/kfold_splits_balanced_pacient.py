import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_patient_based_kfold_splits(
    data_path='',
    num_splits=5,
    random_state=42,
    filename='filename.csv',
    output_filename='kfold_splits_patient.csv',
    counts_filename='kfold_counts_patient.csv',
    label_col='label',
    patient_col='sample_number',
    samples_per_class=10
):
    # 1. Load your dataset
    data = pd.read_csv(os.path.join(data_path, filename))

    # 2. Check columns
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    if patient_col not in data.columns:
        raise ValueError(f"Patient/sample column '{patient_col}' not found.")

    # 3. Create a patient-level dataframe (one row per patient)
    patient_df = data[[patient_col, label_col]].drop_duplicates()

    # 4. Stratified sampling of patients
    neg_patients = patient_df[patient_df[label_col] == 'chipnegative']
    pos_patients = patient_df[patient_df[label_col] == 'chippositive']

    if len(neg_patients) < samples_per_class:
        raise ValueError(f"Not enough negative patients: requested {samples_per_class}, available {len(neg_patients)}.")
    if len(pos_patients) < samples_per_class:
        raise ValueError(f"Not enough positive patients: requested {samples_per_class}, available {len(pos_patients)}.")

    neg_sampled = neg_patients.sample(n=samples_per_class, random_state=random_state)
    pos_sampled = pos_patients.sample(n=samples_per_class, random_state=random_state)

    balanced_patients = pd.concat([neg_sampled, pos_sampled], ignore_index=True)

    # 5. StratifiedKFold on patients
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    balanced_patients['fold_id'] = -1

    for fold_number, (train_idx, test_idx) in enumerate(skf.split(balanced_patients, balanced_patients[label_col])):
        balanced_patients.loc[test_idx, 'fold_id'] = fold_number

    # 6. Merge fold info back to full data
    data = data.merge(balanced_patients[[patient_col, 'fold_id']], on=patient_col, how='inner')

    # 7. Assign split columns (train/val/test) per fold
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        data[fold_col] = 'train'

        test_fold = (i + num_splits - 1) % num_splits
        val_fold  = (i + num_splits - 2) % num_splits

        data.loc[data['fold_id'] == val_fold,  fold_col] = 'val'
        data.loc[data['fold_id'] == test_fold, fold_col] = 'test'

    # 8. Save counts per fold
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
    counts_df.to_csv(os.path.join(data_path, counts_filename), index=False)

    # 9. Remove fold_id if not needed
    data.drop(columns=['fold_id'], inplace=True)

    # 10. Save final CSV
    data.to_csv(os.path.join(data_path, output_filename), index=False)

    print(f"\nPatient-grouped {num_splits}-fold splits saved to '{output_filename}'")
    print(f"K-Fold counts saved to '{counts_filename}'")

# Example usage:
in_file = "features_40_dinobloom-g.csv"
out_file = "features_40_dinobloom-g_kfold_pacient.csv"
out_counts = "features_40_dinobloom-g_kfold_counts_pacient.csv"

data_path = '/ictstr01/home/aih/laia.mana/project/DATA/feature_extraction/'
generate_patient_based_kfold_splits(
    data_path=data_path,
    num_splits=5,
    random_state=42,
    filename=in_file,
    output_filename=out_file,
    counts_filename=out_counts,
    samples_per_class=12
)
