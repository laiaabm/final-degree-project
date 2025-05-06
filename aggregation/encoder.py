def encoder_ndim(encoder):
    if encoder.startswith('tres50'):
        ndim = 1024
    elif encoder == 'ctranspath':
        ndim = 768
    elif encoder == 'phikon':
        ndim = 768
    elif encoder == 'uni':
        ndim = 1024
    elif encoder == 'uni2':
        ndim = 1536
    elif encoder == 'virchow':
        ndim = 2560
    elif encoder == 'virchow2':
        ndim = 2560
    elif encoder == 'dinobloom-s':
        ndim = 384
    elif encoder == 'dinobloom-g':
        ndim = 1536
    elif encoder == 'gigapath':
        ndim = 1536
    elif encoder.startswith('dinosmall'):
        ndim = 384
    elif encoder.startswith('dinobase'):
        ndim = 768
    return ndim