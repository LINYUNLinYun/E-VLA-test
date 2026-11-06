"""Preregistered case252-only 2D RGB mother-texture resolution adaptation."""
import numpy as np
from PIL import Image


def auxiliary_resample(source,target_shape,texture_type):
    source=np.asarray(source)
    if source.dtype!=np.uint8 or source.shape!=(1024,1024,3):
        raise ValueError('Only the registered1024x1024 uint8 RGB mother texture is supported')
    if tuple(target_shape)!=(512,512,3) or int(texture_type)!=0:
        raise ValueError('Only registered512x512 RGB 2D texture target is supported; no cube adaptation')
    return np.asarray(Image.fromarray(source).resize((512,512),resample=Image.Resampling.LANCZOS)).copy()
