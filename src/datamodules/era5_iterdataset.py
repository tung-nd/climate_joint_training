import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class NpyReader(IterableDataset):
    def __init__(self, file_list, variables, out_variables, shuffle: bool = False) -> None:
        super().__init__()
        self.file_list_inp = [f for f in file_list if 'inp' in f]
        self.file_list_out = [f for f in file_list if 'out' in f]
        assert len(self.file_list_inp) == len(self.file_list_out)
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            index_shuf = list(range(len(self.file_list_inp)))
            random.shuffle(index_shuf)
            file_list_inp_shuf = []
            file_list_out_shuf = []
            for i in index_shuf:
                file_list_inp_shuf.append(self.file_list_inp[i])
                file_list_out_shuf.append(self.file_list_out[i])
            self.file_list_inp = file_list_inp_shuf
            self.file_list_out = file_list_out_shuf
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list_inp) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            # iter_end = min(iter_start + per_worker, len(self.file_list))
            iter_end = iter_start + per_worker

        # print(f"rank {rank}")
        # print(f"world size {world_size}")
        # print(f"len data {len(self.file_list)}")
        # print(f"start {iter_start}, end {iter_end}")

        # count the number of data points this worker holds
        # num_data = 0
        # for idx in range(iter_start, iter_end):
        #     data = np.load(self.file_list[idx])
        #     num_data += data["t2m"].shape[0]

        # print(f"rank {rank}")
        # print(f"{num_data} data points")

        # print("==============================")

        for idx in range(iter_start, iter_end):
            path_inp = self.file_list_inp[idx]
            path_out = self.file_list_out[idx]
            data_inp = np.load(path_inp)
            data_out = np.load(path_out)
            yield {k: data_inp[k] for k in self.variables}, {k: data_out[k] for k in self.out_variables}, self.variables, self.out_variables


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, subsample: int = 1
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.subsample = subsample

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data_inp, data_out, variables, out_variables in self.dataset:
            inp = np.concatenate([data_inp[k].astype(np.float32) for k in data_inp.keys()], axis=1)
            inp = torch.from_numpy(inp)
            out = np.concatenate([data_out[k].astype(np.float32) for k in data_out.keys()], axis=1)
            out = torch.from_numpy(out)

            inp = inp[:: self.subsample]
            out = out[:: self.subsample]

            yield inp, out, variables, out_variables


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset: Forecast, transforms: torch.nn.Module, output_transforms: torch.nn.Module):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms

    def __iter__(self):
        for (inp, out, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                if self.transforms is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), variables, out_variables
                else:
                    yield inp[i], out[i], variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
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
        # try:
        #     dataset_iter = iter(self.dataset)
        #     for i in range(self.buffer_size):
        #         shufbuf.append(next(dataset_iter))
        # except:
        #     self.buffer_size = len(shufbuf)

        # try:
        #     while True:
        #         try:
        #             item = next(dataset_iter)
        #             evict_idx = random.randint(0, self.buffer_size - 1)
        #             yield shufbuf[evict_idx]
        #             shufbuf[evict_idx] = item
        #         except StopIteration:
        #             break
        #     while len(shufbuf) > 0:
        #         yield shufbuf.pop()
        # except GeneratorExit:
        #     pass


# x = torch.randn((10, 2))
# pred_range = 2
# history = 3
# interval = 2
# pred_steps = 2
# subsample = 3

# inputs = x.unsqueeze(0).repeat_interleave(history, dim=0)
# for t in range(history):
#     inputs[t] = inputs[t].roll(-t*interval, dims=0)

# # forecast training dataset
# last_idx = -((history - 1) * interval + pred_range)

# outputs = x.roll(last_idx, dims=0)

# inputs = inputs[:, :last_idx].transpose(0, 1)
# outputs = outputs[:last_idx]

# inputs = inputs[::subsample]
# outputs = outputs[::subsample]

# # forecast validation dataset
# outputs = x.unsqueeze(0).repeat_interleave(pred_steps, dim=0)
# start_idx = (history-1) * interval + pred_range
# for t in range(pred_steps):
#     outputs[t] = outputs[t].roll(-(start_idx + t*pred_range), dims=0)

# last_idx = - ((history-1) * interval + pred_steps * pred_range)

# inputs = inputs[:, :last_idx].transpose(0, 1)
# outputs = outputs[:, :last_idx].transpose(0, 1)

# for i in range(inputs.shape[0]):
#     print ('x', x)
#     print (i)
#     print ('in', inputs[i])
#     print ('out', outputs[i])
#     print ('=' * 20)

# import os

# import torchdata.datapipes as dp

# root_dir = "/datadrive/datasets/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualForecastDataIter(
#         dataset=ERA5Forecast(
#             dataset=ERA5Npy(
#                 file_list=lister_train,
#                 variables=["t2m", "u10", "v10", "z_500", "t_850"],
#                 out_variables=["z_500", "t_850"],
#             ),
#             predict_range=6,
#             history=3,
#             interval=6
#         ),
#         transforms=None,
#         output_transforms=None
#     ),
#     buffer_size=1000,
# )

# x, y, variables, out_variables = next(iter(dataset))
# print(x.shape)
# print(y.shape)
# print (variables)
# print (out_variables)

# root_dir = "/datadrive/datasets/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualForecastDataIter(
#         dataset=ERA5ForecastMultiStep(
#             dataset=ERA5Npy(
#                 file_list=lister_train,
#                 variables=["t2m", "u10", "v10", "z_500", "t_850"],
#                 out_variables=None
#             ),
#             pred_range=6,
#             history=3,
#             interval=6,
#             pred_steps=4,
#         ),
#         transforms=None,
#         output_transforms=None
#     ),
#     buffer_size=1000,
# )

# x, y, variables, out_variables = next(iter(dataset))
# print(x.shape)
# print(y.shape)
# print (variables)
# print (out_variables)

# root_dir = "/datadrive/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualDataIter(
#         dataset=ERA5(
#             dataset=ERA5Npy(
#                 file_list=lister_train, variables=["t2m", "u10", "v10", "z"]
#             ),
#         ),
#         transforms=None,
#     ),
#     buffer_size=1000,
# )

# x, variables = next(iter(dataset))
# print(x.shape)
# print (variables)


# root_dir = "/datadrive/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualDataIter(
#         dataset=ERA5Video(
#             dataset=ERA5Npy(
#                 file_list=lister_train, variables=["t2m", "u10", "v10", "z"]
#             ), timesteps=4
#         ),
#         transforms=None,
#     ),
#     buffer_size=1000,
# )

# x, variables = next(iter(dataset))
# print(x.shape)
# print (variables)

# x = y = torch.rand(10, 2)
# history = 3
# interval = 1
# max_predict_range = 3
# inputs = x.unsqueeze(0).repeat_interleave(history, dim=0)
# for t in range(history):
#     inputs[t] = inputs[t].roll(-t * interval, dims=0)

# last_idx = -((history - 1) * interval + max_predict_range)

# inputs = inputs[:, :last_idx].transpose(0, 1)  # N, T, C, H, W

# random_predict_ranges = torch.randint(low=1, high=max_predict_range, size=(inputs.shape[0],))
# output_ids = torch.arange(inputs.shape[0]) + (history - 1) * interval + random_predict_ranges
# outputs = y[output_ids]

# print ('data', x)
# print ('inputs', inputs)
# print ('predict ranges', random_predict_ranges)
# print ('outputs', outputs)
