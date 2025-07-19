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

def mean_relative_error(pred, lbl):
    pred = pred.detach().cpu().numpy()
    lbl = lbl.detach().cpu().numpy()
    return np.mean(np.abs(pred - lbl) / lbl)

def root_mean_squared_error(pred, lbl):
    pred = pred.detach().cpu().numpy()
    lbl = lbl.detach().cpu().numpy()
    return np.sqrt(np.mean(np.square(pred - lbl)))

def average_log_error(pred, lbl):
    pred = pred.detach().cpu().numpy()
    lbl = lbl.detach().cpu().numpy()
    return np.mean(np.abs(np.log(pred) - np.log(lbl)))