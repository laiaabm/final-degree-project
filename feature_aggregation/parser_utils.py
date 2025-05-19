import argparse

def args_parser_test():
    parser = argparse.ArgumentParser()

    # I/O PARAMS
    parser.add_argument('--output', type=str, default='.', help='name of output directory')
    parser.add_argument('--data', type=str, default='', help='which data to use')
    parser.add_argument('--encoder', type=str, default='', choices=[
        'tres50_imagenet',
        'ctranspath',
        'phikon',
        'uni',
        'uni2',
        'virchow',
        'virchow2',
        'dinobloom-s',
        'dinobloom-g',
        'gigapath',
        'dinosmall',
        'dinobase'
    ], help='which encoder to use')
    parser.add_argument('--method', type=str, default='', choices=[
        'AB-MIL',
        'AB-MIL_FC_small',
        'AB-MIL_FC_big',
        'CLAM_SB',
        'CLAM_MB',
        'transMIL',
        'DS-MIL',
        'VarMIL',
        'GTP',
        'PatchGCN',
        'DeepGraphConv',
        'ViT_MIL',
        'DTMIL',
        'LongNet_ViT'
    ], help='which aggregation method to use')
    parser.add_argument('--kfold', default=0, type=int, choices=list(range(0,5)), help='which fold (0 to 5)')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--checkpoint', type=str, default='', help='path to the model checkpoint')
    parser.add_argument('--log_csv', type=str, default='inference_results.csv', help='CSV file to save the results')

    # OPTIMIZATION PARAMS
    parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')
    
    return parser


def args_parser_train():    
    parser = argparse.ArgumentParser()

    # I/O PARAMS
    parser.add_argument('--output', type=str, default='.', help='name of output directory')
    parser.add_argument('--data', type=str, default='', help='which data to use')
    parser.add_argument('--encoder', type=str, default='', choices=[
        'tres50_imagenet',
        'ctranspath',
        'phikon',
        'uni',
        'uni2',
        'virchow',
        'virchow2',
        'dinobloom-s',
        'dinobloom-g',
        'gigapath',
        'dinosmall',
        'dinobase'
    ], help='which encoder to use')
    parser.add_argument('--method', type=str, default='', choices=[
        'AB-MIL',
        'AB-MIL_FC_small',
        'AB-MIL_FC_big',
        'CLAM_SB',
        'CLAM_MB',
        'transMIL',
        'Transformer',
        'Transformer2',
        'ViT',
        'DS-MIL',
        'VarMIL',
        'GTP',
        'PatchGCN',
        'DeepGraphConv',
        'ViT_MIL',
        'DTMIL',
        'LongNet_ViT'
    ], help='which aggregation method to use')
    parser.add_argument('--kfold', default=0, type=int, choices=list(range(0,5)), help='which 0 t0 5?')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')

    # OPTIMIZATION PARAMS
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training).""")
    parser.add_argument('--lr_end', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay.""")
    parser.add_argument('--nepochs', type=int, default=50, help='number of epochs (default: 40)')
    parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed')

    return parser