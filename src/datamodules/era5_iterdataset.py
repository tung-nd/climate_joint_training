import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset
from typing import Union


def shuffle_two_list(list1, list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(list1)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf, list2_shuf


class NpyReader(IterableDataset):
    def __init__(
        self,
        inp_file_list,
        out_file_list,
        variables,
        out_variables,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        assert len(inp_file_list) == len(out_file_list)
        self.inp_file_list = [f for f in inp_file_list if 'climatology' not in f]
        self.out_file_list = [f for f in out_file_list if 'climatology' not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.inp_file_list, self.out_file_list = shuffle_two_list(
                self.inp_file_list, self.out_file_list
            )

        n_files = len(self.inp_file_list)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = n_files
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(n_files / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path_inp = self.inp_file_list[idx]
            path_out = self.out_file_list[idx]
            inp = np.load(path_inp)
            if path_out == path_inp:
                out = inp
            else:
                out = np.load(path_out)
            yield {k: inp[k] for k in self.variables}, {
                k: out[k] for k in self.out_variables
            }, self.variables, self.out_variables


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, pred_range: int = 6, history: int = 3, window: int = 6
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.pred_range = pred_range
        self.history = history
        self.window = window

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            x = np.concatenate(
                [inp_data[k].astype(np.float32) for k in inp_data.keys()], axis=1
            )
            x = torch.from_numpy(x)
            y = np.concatenate(
                [out_data[k].astype(np.float32) for k in out_data.keys()], axis=1
            )
            y = torch.from_numpy(y)

            inputs = x.unsqueeze(0).repeat_interleave(self.history, dim=0)
            for t in range(self.history):
                inputs[t] = inputs[t].roll(-t * self.window, dims=0)

            last_idx = -((self.history - 1) * self.window + self.pred_range)

            inputs = inputs[:, :last_idx].transpose(0, 1)  # N, T, C, H, W

            predict_ranges = (
                torch.ones(inputs.shape[0]).to(torch.long) * self.pred_range
            )
            output_ids = (
                torch.arange(inputs.shape[0])
                + (self.history - 1) * self.window
                + predict_ranges
            )
            outputs = y[output_ids]

            yield inputs, outputs, variables, out_variables


class Downscale(IterableDataset):
    def __init__(self, dataset: NpyReader) -> None:
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            x = np.concatenate(
                [inp_data[k].astype(np.float32) for k in inp_data.keys()], axis=1
            )
            x = torch.from_numpy(x)
            y = np.concatenate(
                [out_data[k].astype(np.float32) for k in out_data.keys()], axis=1
            )
            y = torch.from_numpy(y)

            yield x.unsqueeze(1), y, variables, out_variables
            
            
class Joint(IterableDataset):
    def __init__(
        self, dataset: NpyReader, pred_range: int = 6, history: int = 3, window: int = 6
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.pred_range = pred_range
        self.history = history
        self.window = window
        
    def __iter__(self):
        # inp_data: low-res, out_data: high-res
        for inp_data, out_data, variables, out_variables in self.dataset:
            x = np.concatenate(
                [inp_data[k].astype(np.float32) for k in inp_data.keys()], axis=1
            )
            x = torch.from_numpy(x)
            y = np.concatenate(
                [out_data[k].astype(np.float32) for k in out_data.keys()], axis=1
            )
            y = torch.from_numpy(y)
            
            # input of low-res forecasting
            inputs = x.unsqueeze(0).repeat_interleave(self.history, dim=0)
            for t in range(self.history):
                inputs[t] = inputs[t].roll(-t * self.window, dims=0)

            last_idx = -((self.history - 1) * self.window + self.pred_range)

            inputs = inputs[:, :last_idx].transpose(0, 1)  # N, T, C, H, W
            
            # output of low-res forecasting
            predict_ranges = (
                torch.ones(inputs.shape[0]).to(torch.long) * self.pred_range
            )
            output_ids = (
                torch.arange(inputs.shape[0])
                + (self.history - 1) * self.window
                + predict_ranges
            )
            forecast_outputs = x[output_ids]
            
            # output of downscaling
            downscale_outputs = y[output_ids]
            
            yield inputs, forecast_outputs, downscale_outputs, variables, out_variables


class IndividualDataIter(IterableDataset):
    def __init__(
        self,
        dataset: Union[Forecast, Downscale],
        transforms: torch.nn.Module,
        output_transforms: torch.nn.Module,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms

    def __iter__(self):
        for inp, out, variables, out_variables in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                if self.transforms is not None:
                    yield self.transforms(inp[i]), self.output_transforms(
                        out[i]
                    ), variables, out_variables
                else:
                    yield inp[i], out[i], variables, out_variables
                    

class IndividualJointDataIter(IterableDataset):
    def __init__(
        self,
        dataset: Union[Forecast, Downscale],
        transforms: torch.nn.Module,
        forecast_output_transforms: torch.nn.Module,
        downscale_output_transforms: torch.nn.Module,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.forecast_output_transforms = forecast_output_transforms
        self.downscale_output_transforms = downscale_output_transforms

    def __iter__(self):
        for inp, forecast_out, downscale_out, variables, out_variables in self.dataset:
            assert inp.shape[0] == forecast_out.shape[0] and forecast_out.shape[0] == downscale_out.shape[0]
            for i in range(inp.shape[0]):
                if self.transforms is not None:
                    yield (
                        self.transforms(inp[i]),
                        self.forecast_output_transforms(forecast_out[i]),
                        self.downscale_output_transforms(downscale_out[i]),
                        variables, out_variables
                    )
                else:
                    yield inp[i], forecast_out[i], downscale_out[i], variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset: IndividualDataIter, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()
