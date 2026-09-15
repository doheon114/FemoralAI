from .counterfactual_inpainting_cgan_femoral import CounterfactualInpaintingCGANFemoral


def build_gan(opt, **kwargs):
    assert 'kind' in opt, 'No architecture type specified in the model configuration'
    if opt.kind == 'inpainting_counterfactual_cgan_femoral':
        return CounterfactualInpaintingCGANFemoral(opt=opt, **kwargs)
    else:
        raise ValueError(f'Invalid architecture type: {opt.kind}')
