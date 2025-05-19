# Import aggregation methods
from aggregator import GMA, TransMIL

def get_aggregator(method='', ndim=1024, n_classes=2, **kwargs):
    # GMA
    if method == 'AB-MIL':
        return GMA(ndim=ndim, n_classes=n_classes, **kwargs)
    # TransMIL
    elif method == 'transMIL':
        return TransMIL(ndim=ndim, n_classes=n_classes, **kwargs)
    else:
        raise Exception(f'Method {method} not defined')
