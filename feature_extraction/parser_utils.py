import argparse

def args_parser():      
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the tile folders (e.g. IC1_24_1a, IC2_24_1b, etc.)'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to a CSV file containing columns: slide_id, label'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='Path to store the extracted features (.h5 files)'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        choices=[
            'uni2',
            'dinobloom-s',
            'dinobloom-g',
        ],
        help='Choice of encoder architecture'
    )
    parser.add_argument('--tilesize', type=int, default=224, help='Tile size for resizing patches')

    return parser