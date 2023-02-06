import torch
from torch import nn
from src.datamodules import CONSTANTS, NAME_TO_VAR


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def get_constant_ids(self, all_ids, vars):
        constant_ids = []
        for id in all_ids:
            var = vars[id]
            if var in NAME_TO_VAR:
                if NAME_TO_VAR[var] in CONSTANTS:
                    constant_ids.append(id)
        return constant_ids
    
    def replace_constant_variables(self, inp, pred, in_vars, out_vars):
        # do not predict constant variables
        # replace prediction of constant variables by the ground truth
        out_var_id_constants = self.get_constant_ids(range(pred.shape[-3]), out_vars)
        if len(out_var_id_constants) > 0:
            in_var_id_constants = self.get_constant_ids(range(inp.shape[-3]), in_vars)
            assert len(in_var_id_constants) == len(out_var_id_constants)
            pred[:, out_var_id_constants] = inp[:, 0, in_var_id_constants].to(pred.dtype)
        return pred
    
    def predict(self, inp, in_vars, out_vars):
        pass

    def forward(self, x: torch.Tensor, y: torch.Tensor, variables, out_variables, metric, lat, return_pred=False):
        # B, C, H, W
        pred = self.predict(x, variables, out_variables)
        if return_pred:
            return [m(pred, y, out_variables, lat) for m in metric], x, pred
        else:
            return [m(pred, y, out_variables, lat) for m in metric], x
        
    def evaluate(self, x, y, variables, out_variables, transform, metrics, lat, clim, log_postfix, return_pred=False):
        pred = self.predict(x, variables, out_variables)
        if return_pred:
            return [m(pred, y, transform, out_variables, lat, clim, log_postfix) for m in metrics], pred
        else:
            return [m(pred, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]