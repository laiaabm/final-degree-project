def encoder_ndim(encoder):
    if encoder == 'uni':
        ndim = 1024
    elif encoder == 'uni2':
        ndim = 1536
    elif encoder == 'dinobloom-s':
        ndim = 384
    elif encoder == 'dinobloom-g':
        ndim = 1536

    return ndim