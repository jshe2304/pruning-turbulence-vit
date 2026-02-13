from src.models.vit import ViT
from src.models.re_vit import ReViT
from src.models.encoder_decoder_vit import EncoderDecoderViT
from src.models.unet import UNet

MODEL_REGISTRY = {
    'ViT': ViT,
    'ReViT': ReViT,
    'EncoderDecoderViT': EncoderDecoderViT,
    'UNet': UNet,
}

def create_model(model_type, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)
