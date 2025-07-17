import torch
import numpy as np

def model_in_out_shape(model, input_shape=[1, 1, 100, 100], DEVICE="cuda"):
    """
    input_shape is in (B, C, H, W) format
    """
    model.to(DEVICE)
    silly_img = torch.from_numpy(np.random.randn(*input_shape).astype("float32")).to(DEVICE)
    silly_pred = model(silly_img)

    return silly_pred.shape