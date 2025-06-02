import torchvision.transforms as transforms

def get_encoder(encoder):
    if encoder == 'uni2':
        from encoder import uni
        model = uni.get_model()
        ndim = 1536
    elif encoder == 'dinobloom-s':
        from encoder import dinobloom
        model_name="dinov2_vits14_S"
        model = dinobloom.DinoBloom(model_name)
        ndim = 384
    elif encoder == 'dinobloom-g':
        from encoder import dinobloom
        model_name="dinov2_vitg14_G"
        model = dinobloom.DinoBloom(model_name)
        ndim = 1536
    else:
        raise Exception('Wrong encoder name')
    
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                                ])
    total_params = sum(p.numel() for p in model.parameters())
    # print(f'Model: {encoder} - {total_params} parameters')
    return model, transform, ndim
