import pandas as pd
import h5py

def get_datasets_kfold(kfold=0, data='', encoder='', label_dict=None):
    if label_dict is None:
        # Provide a default or raise an error if you expect a label_dict
        label_dict = {'chipnegative': 0, 'chippositive': 1}

    # Load slide data
    df = pd.read_csv(data)
    # clean data
    df = df.rename(columns={
        'patient_id': 'sample_number', 
        'label': 'target', 
        'kfold{}'.format(kfold): 'kfoldsplit', 
        'tensor_paths': 'tensor_path'
    })[['sample_number','target','kfoldsplit','tensor_path']]

    # Split into train and val
    df_train = df[df.kfoldsplit == 'train'].reset_index(drop=True).drop(columns=['kfoldsplit'])
    df_val   = df[df.kfoldsplit == 'val'].reset_index(drop=True).drop(columns=['kfoldsplit'])
    df_test  = df[df.kfoldsplit == 'test'].reset_index(drop=True).drop(columns=['kfoldsplit'])

    # Create your dataset objects, passing in label_dict
    dset_train = slide_dataset_classification(df_train, label_dict)
    dset_val   = slide_dataset_classification(df_val, label_dict)
    dset_test  = slide_dataset_classification(df_test, label_dict)

    return dset_train, dset_val, dset_test

class slide_dataset_classification(object):
    '''
    Slide-level dataset which returns, for each slide, the feature matrix (h) and the target
    '''
    def __init__(self, df, label_dict):
        self.df = df
        self.label_dict = label_dict
    
    def __len__(self):
        # number of slides
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Load the feature matrix for that slide from the .h5 file
        with h5py.File(row.tensor_path, 'r') as file:
            feat = file['features'][:]
        # Map the label to its corresponding integer value using label_dict
        target = self.label_dict[row.target]
        
        return feat, target
