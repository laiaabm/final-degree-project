import datasets
import modules
import os
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from parser_utils import args_parser_train
from encoder import encoder_ndim


def set_random_seed(seed_value):
    random.seed(seed_value) # Python random module
    np.random.seed(seed_value) # Numpy module
    torch.manual_seed(seed_value) # Sets the seed for generating random numbers for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value) # Sets the seed for generating random numbers on all GPUs
        torch.cuda.manual_seed_all(seed_value) # Sets the seed for generating random numbers on all GPUs
        torch.backends.cudnn.deterministic = True # Makes CUDA operations deterministic
        torch.backends.cudnn.benchmark = False


def train(run, loader, model, criterion, optimizer, lr_schedule, wd_schedule):
    # Set model in train mode
    model.train()
    running_loss = 0.

    # Loop through batches
    for i, (input, target) in enumerate(loader):
        # Update weight decay and learning rate
        it = len(loader) * (run-1) + i # global training iteration
        for j, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
            if j == 0: # only the first group is regularized
                param_group['weight_decay'] = wd_schedule[it]
        
        feat = input.float().squeeze(0).cuda()
        target = target.long().cuda()
        
        # Forward pass
        results_dict = model(feat)
        logits = results_dict['logits']
        loss = criterion(logits, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        print(f'Training\tEpoch: [{run}/{args.nepochs}]\tBatch: [{i+1}/{len(loader)}]\tLoss: {loss.item()}')

    return running_loss / len(loader)


def test(run, loader, model, args):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader), args.num_classes).cuda()
    # Initialize a list to store probabilities, true labels, and other details for each image
    results = []

    # Loop through batches
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            print(f'Inference\tEpoch: [{run}/{args.nepochs}]\tBatch: [{i+1}/{len(loader)}]')

            # Copy batch to GPU
            feat = input.float().squeeze(0).cuda()
            # Forward pass through the model
            results_dict = model(feat)
            logits, Y_prob, Y_hat = (results_dict[key] for key in ['logits', 'Y_prob', 'Y_hat']) 

            # Squeeze if needed
            if Y_prob.dim() > 1 and Y_prob.size(1) == 1:
                Y_prob = Y_prob.squeeze(1)

            probs[i] = Y_prob.detach()
            results.append({
                'patient_id': loader.dataset.df.iloc[i]['sample_number'],  # Adjust column name if needed
                'true_label': loader.dataset.df.iloc[i]['target'],
                'probabilities': Y_prob.detach().cpu().numpy().squeeze()
            })

    proabilities = probs.cpu().numpy()
    results_df = pd.DataFrame(results)
    return proabilities, results_df


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith('.bias') or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def main():
    # Get user input
    global args
    parser = args_parser_train()
    args = parser.parse_args()

    if args.random_seed:
        set_random_seed(args.random_seed)
    
    # Create a subdirectory for outputs based on the aggregation method
    args.save_dir = os.path.join(args.output, args.method)
    os.makedirs(args.save_dir, exist_ok=True) # ensure directory exists
    
    # Now store the log file in this subdirectory
    log_path = os.path.join(args.save_dir, f'convergence_kfold{args.kfold}.csv')
    
    # Label dictionary (example for a certain dataset)
    label_dict = {'chipnegative': 0, 'chippositive': 1} 
    
    # Prepare datasets
    train_dset, val_dset, test_dset = datasets.get_datasets_kfold(kfold=args.kfold, data=args.data, encoder=args.encoder, label_dict=label_dict)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    # Dim of features based on encoder
    ndim = encoder_ndim(args.encoder)
    
    # Get model
    model = modules.get_aggregator(method=args.method, ndim=ndim, n_classes=args.num_classes)
    model.cuda()
    
    # Set loss
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Set optimizer
    params_groups = get_params_groups(model)
    optimizer = optim.AdamW(params_groups)
    
    # Set schedulers
    lr_schedule = cosine_scheduler(args.lr, args.lr_end, args.nepochs, len(train_loader), warmup_epochs=args.warmup_epochs)
    wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.nepochs, len(train_loader))
    cudnn.benchmark = True
    
    # Initialize logs
    with open(log_path, 'w') as fconv:
        fconv.write('epoch,metric,value\n')
    
    # Track the best AUC, best F1, and best Accuracy
    best_auc = 0.0
    best_f1 = 0.0
    best_acc = 0.0
    
    # Main training loop
    for epoch in range(args.nepochs+1):
        
        # Training
        if epoch > 0:
            loss = train(epoch, train_loader, model, criterion, optimizer, lr_schedule, wd_schedule)
            print(f'Training\tEpoch: [{epoch}/{args.nepochs}]\tLoss: {loss}')

            with open(log_path, 'a') as fconv:
                fconv.write(f'{epoch}, loss, {loss}\n')
        
        # Validation
        probs, results_df = test(epoch, val_loader, model, args)
        
        # Convert the ground-truth to numeric array
        y_true = val_loader.dataset.df['target'].map(label_dict).values

        # AUC score
        pos_probs = probs[:, 1] # since class 1 is the 'positive' class
        auc = roc_auc_score(y_true, pos_probs)
        
        # F1 score (macro); get predicted labels
        y_pred = np.argmax(probs, axis=1)
        f1 = f1_score(y_true, y_pred, average='binary')

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Check if the current model is best by AUC
        if auc > best_auc:
            print(f'New best AUC model found at epoch {epoch} with AUC: {auc}')
            best_auc = auc
            # Save checkpoint
            obj = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'auc': auc,
                'f1': f1,
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            ### Save best checkpoint (by AUC)
            torch.save(obj, os.path.join(args.save_dir, f'checkpoint_best_auc_kfold{args.kfold}.pth'))
        
        # Check if the current model is best by F1
        if f1 > best_f1:
            print(f'New best F1 model found at epoch {epoch} with F1: {f1}')
            best_f1 = f1
            # Save checkpoint
            obj = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'auc': auc,
                'f1': f1,
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            ### Save best checkpoint (by F1)
            torch.save(obj, os.path.join(args.save_dir, f'checkpoint_best_f1_kfold{args.kfold}.pth'))
        
        # Check if the current model is best by Accuracy
        if acc > best_acc:
            print(f'New best Accuracy model found at epoch {epoch} with Accuracy: {acc}')
            best_acc = acc
            # Save checkpoint
            obj = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'auc': auc,
                'f1': f1,
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            ### Save best checkpoint (by Accuracy)
            torch.save(obj, os.path.join(args.save_dir, f'checkpoint_best_acc_kfold{args.kfold}.pth'))

        # Print stats
        print(f'Validation\tEpoch: [{epoch}/{args.nepochs}]\tAUC: {auc:.4f}\tF1: {f1:.4f}\tAcc: {acc:.4f}')

        with open(log_path, 'a') as fconv:
            fconv.write(f'{epoch}, auc, {auc}\n')
            fconv.write(f'{epoch}, f1, {f1}\n')
            fconv.write(f'{epoch}, acc, {acc}\n')
        
        # Always save a 'latest' checkpoint if you want
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'auc': auc,
            'f1': f1,
            'acc': acc,
            'optimizer': optimizer.state_dict()
        }
        torch.save(obj, os.path.join(args.save_dir, f'checkpoint_latest_kfold{args.kfold}.pth'))

    # Build the confusion matrix
    true_labels = results_df['true_label'].map(label_dict).values
    predicted_labels = np.argmax(probs, axis=1)
    reverse_label_dict = {v: k for k, v in label_dict.items()} # Reverse mapping for easy lookup

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=list(reverse_label_dict.values()), columns=list(reverse_label_dict.values()))
    conf_matrix_df.to_csv(os.path.join(args.save_dir, '{}_confusion_matrix_kf{}.csv'.format(args.encoder, args.kfold)), index=True)


if __name__ == '__main__':
    main()