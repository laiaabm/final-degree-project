import datasets
import modules
import os
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from parser_utils import args_parser_test
from encoder import encoder_ndim


def test(run, loader, model, args):
    # Set model in test mode
    model.eval()

    # Initialize a list to store probabilities, true labels, and other details for each image
    results = []
    # Inintialize a list to store the attention scores
    attn_scores = []

    # Loop through batches
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            print(f'Inference\tBatch: [{i+1}/{len(loader)}]')
            
            # Copy batch to GPU
            feat = input.float().squeeze(0).cuda()

            # Forward pass through the model
            results_dict = model(feat)
            logits, Y_prob, Y_hat, attn_scores_layer1, attn_scores_layer2 = (results_dict[key] for key in ['logits', 'Y_prob', 'Y_hat', 'attn_scores_layer1', 'attn_scores_layer2'])  # Include A

            # results stores the patient ID, probabilities and true label
            results.append({
                'patient_id': loader.dataset.df.iloc[i]['sample_number'], # Assuming that 'sample_number' holds the sample ID
                'true_label': loader.dataset.df.iloc[i]['target'], # True label
                'probabilities': Y_prob.detach().cpu().numpy().squeeze() # Predicted probabilities
            })

            # attn_scores stores the attention scores as well as the patient ID
            attn_scores.append({
                'patient_id': loader.dataset.df.iloc[i]['sample_number'],
                'attn_scores_layer1': attn_scores_layer1.detach().cpu().numpy().squeeze(),
                'attn_scores_layer2': attn_scores_layer2.detach().cpu().numpy().squeeze()
            })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
    attention_df = pd.DataFrame(attn_scores)
    return results_df, attention_df


def main():
    # Get user input
    global args
    parser = args_parser_test()
    args = parser.parse_args()

    args.save_dir = os.path.join(args.output, args.method)
    os.makedirs(args.save_dir, exist_ok=True) # ensure directory exists

    # Load datasets (train_dset is not used for inference)
    label_dict = {'chipnegative': 0, 'chippositive': 1} 
    train_dset, val_dset, test_dset = datasets.get_datasets_kfold(kfold=args.kfold, data=args.data, encoder=args.encoder, label_dict=label_dict)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=args.workers)

    # Determine model feature dimensions based on the encoder type
    ndim = encoder_ndim(args.encoder)

    # Get model
    model = modules.get_aggregator(method=args.method, ndim=ndim, n_classes=args.num_classes)
    model.cuda()

    # Load pretrained model checkpoint
    if args.checkpoint:
        print(f'Loading checkpoint from {args.checkpoint}...')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Checkpoint loaded, AUC at checkpoint: {checkpoint.get("auc", "N/A")}')

    # Perform inference on test data and log results
    print('Starting inference...')
    results_df, attention_df = test(0, test_loader, model, args)

    # Get the true labels and probabilities
    reverse_label_dict = {v: k for k, v in label_dict.items()} # Reverse mapping for easy lookup

    # Map true labels, merging 'MCL' and 'MAL' into the same class
    true_labels = results_df['true_label'].map(label_dict).values
    probs = np.stack(results_df['probabilities'].values)
    
    # Predicted labels (argmax to get the predicted class)
    predicted_labels = np.argmax(probs, axis=1)

    # AUC Score
    pos_probs = probs[:, 1] # since class 1 is the 'positive' class
    auc = roc_auc_score(true_labels, pos_probs)
    
    # F1-macro score
    f1_macro = f1_score(true_labels, predicted_labels, average='binary')

    # Accuracy
    acc = accuracy_score(true_labels, predicted_labels)

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Save the results (including per-image probabilities, true labels, predicted labels)
    results_df['predicted_label'] = predicted_labels
    results_df['auc'] = auc
    results_df['f1_macro'] = f1_macro
    results_df['acc'] = acc

    # Save the results to CSV
    results_df.to_csv(os.path.join(args.save_dir, f'{args.encoder}_inference_results_kf{args.kfold}.csv'), index=False)
    attention_df.to_csv(os.path.join(args.save_dir, f'{args.encoder}_attention_scores_kf{args.kfold}.csv'), index=False)

    # Save confusion matrix to CSV as well
    conf_matrix_df = pd.DataFrame(conf_matrix, index=list(reverse_label_dict.values()), columns=list(reverse_label_dict.values()))
    conf_matrix_df.to_csv(os.path.join(args.save_dir, f'{args.encoder}_confusion_matrix_kf{args.kfold}.csv'), index=True)
    
    print(f'Results saved to {args.log_csv} and confusion matrix saved to confusion_matrix.csv')


if __name__ == '__main__':
    main()