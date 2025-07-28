import torch
from torch import nn
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def laplacian_loss(pred, target):
    b, c, h, w = pred.shape

    kernel = torch.tensor([
        [1,  4,  1],
        [4, -20, 4],
        [1,  4,  1]
    ], dtype=pred.dtype, device=pred.device)

    kernel = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)

    lap_pred = F.conv2d(pred, kernel, padding=1, groups=c)
    lap_target = F.conv2d(target, kernel, padding=1, groups=c)

    return F.l1_loss(lap_pred, lap_target)


def scharr_loss(pred, target):
    b, c, h, w = pred.shape

    scharr_x = torch.tensor([
        [-3,  0,  3],
        [-10, 0, 10],
        [-3,  0,  3]
    ], dtype=pred.dtype, device=pred.device)

    scharr_y = torch.tensor([
        [ 3, 10,  3],
        [ 0,  0,  0],
        [-3, -10, -3]
    ], dtype=pred.dtype, device=pred.device)

    scharr_x = scharr_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    scharr_y = scharr_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1)

    grad_x_pred = F.conv2d(pred, scharr_x, padding=1, groups=c)
    grad_y_pred = F.conv2d(pred, scharr_y, padding=1, groups=c)

    grad_x_target = F.conv2d(target, scharr_x, padding=1, groups=c)
    grad_y_target = F.conv2d(target, scharr_y, padding=1, groups=c)

    loss_x = F.l1_loss(grad_x_pred, grad_x_target)
    loss_y = F.l1_loss(grad_y_pred, grad_y_target)

    return loss_x + loss_y


def Loss_fn(pred, lbl, depth=1.0, struct=1.0, grad=1.0, laplacian=1.0, scharr=1.0):
    L1 = nn.L1Loss()

    l_depth = L1(pred, lbl)
    l_ssim = torch.clamp((1 - ssim(pred, lbl, val_range = 1.0)) * 0.5, 0, 1)
    l_grad = gradient_loss(pred, lbl)
    l_laplacian = laplacian_loss(pred, lbl)
    l_scharr = scharr_loss(pred, lbl)

    loss = (
        depth * l_depth + 
        struct * l_ssim + 
        grad * l_grad + 
        laplacian * l_laplacian + 
        scharr * l_scharr
    )

    return loss