import numpy as np
import torch
from scipy import stats


def mse(pred, y, vars, lat=None, mask=None):
    """
    y: [N, 3, H, W]
    pred: [N, L, p*p*3]
    vars: list of variable names
    """
    loss = (pred - y) ** 2

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = loss[:, i].mean()
    
    if mask is not None:
        loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = loss.mean(dim=1).mean()

    return loss_dict


def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
    
    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    lat: H
    """

    error = (pred - y) ** 2  # [B, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2  # [B, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B C, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


### Downscaling metrics

def mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    """
    error = (pred - y) ** 2 # [B, C, H, W]

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"mse_{var}"] = error[:, i].mean()
            
    loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def rmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    """
    pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    error = (pred - y) ** 2  # [N, C, H, W]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"rmse_{var}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i], dim=(-2, -1)))
            )
            
    loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def remove_nans(pred: torch.Tensor, gt: torch.Tensor):
    # pred and gt are two flattened arrays
    pred_nan_ids = torch.isnan(pred) | torch.isinf(pred)
    pred = pred[~pred_nan_ids]
    gt = gt[~pred_nan_ids]

    gt_nan_ids = torch.isnan(gt) | torch.isinf(gt)
    pred = pred[~gt_nan_ids]
    gt = gt[~gt_nan_ids]

    return pred, gt

def pearson(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    """
    pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            # print (var)
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            pred_, y_ = remove_nans(pred_, y_)
            # if len(pred_) == 0:
            #     import sys
            #     sys.exit()
            loss_dict[f"pearsonr_{var}"] = stats.pearsonr(
                pred_.cpu().numpy(),
                y_.cpu().numpy()
            )[0]
            
    loss_dict["pearson"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict

def mean_bias(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    """
    pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"mean_bias_{var}"] = y.mean() - pred.mean()
            
    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


# pred = torch.randn(2, 4, 3, 128, 256).cuda()
# y = torch.randn(2, 4, 3, 128, 256).cuda()
# vars = ["x", "y", "z"]
# print(lat_weighted_rmse(pred, y, vars))
# print(lat_weighted_acc(pred, y, vars))
