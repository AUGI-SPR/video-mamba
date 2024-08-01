"""
    Adapted from https://github.com/yabufarha/ms-tcn
"""

import os
import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer
import pickle


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        print(self.actions_dict)
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

        self.timewarp_layer = TimeWarpLayer()

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        self.list_of_examples = sorted(
            [f for f in os.listdir(vid_list_file) if f.endswith(".mp4")]
        )
        ##file_ptr = open(vid_list_file, 'r')
        ##self.list_of_examples = file_ptr.read().split('\n')[:-1]
        ##file_ptr.close()
        self.gts = [
            os.path.join(self.gt_path, vid.split(".")[0] + ".txt")
            for vid in self.list_of_examples
        ]
        self.features = [
            self.features_path + vid.split(".")[0] + ".pkl"
            for vid in self.list_of_examples
        ]
        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)

    def warp_video(self, batch_input_tensor, batch_target_tensor):
        """
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        """
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(
            batch_input_tensor, grid, mode="bilinear"
        )
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(
            batch_target_tensor, grid, mode="nearest"
        )  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(
            1
        ).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        """
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        """

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print("Merge! Dataset length:{}".format(len(self.list_of_examples)))

    def next_batch(
        self, batch_size, if_warp=False
    ):  # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index : self.index + batch_size]
        batch_gts = self.gts[self.index : self.index + batch_size]
        batch_features = self.features[self.index : self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx], allow_pickle=True)
            file_ptr = open(batch_gts[idx], "r")
            content = file_ptr.read().split("\n")[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            feature = features[:, :: self.sample_rate]
            target = classes[:: self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)

        length_of_sequences = list(map(len, batch_target))
        max_length = max(length_of_sequences)

        # Initialize tensors with padding
        C_in = np.shape(batch_input[0])[0]
        batch_input_tensor = torch.zeros(
            len(batch_input), C_in, max_length, dtype=torch.float
        )
        batch_target_tensor = torch.ones(
            len(batch_input), max_length, dtype=torch.long
        ) * (
            -100
        )  # Assuming -100 is used for padding
        mask = torch.zeros(
            len(batch_input), self.num_classes, max_length, dtype=torch.float
        )

        # Populate tensors with padded/truncated data
        for i in range(len(batch_input)):
            tensor_input = torch.from_numpy(batch_input[i]).squeeze(-1).squeeze(-1)
            sequence_length = batch_input[i].shape[1]
            batch_input_tensor[i, :, :sequence_length] = tensor_input

            target_length = batch_target[i].shape[0]
            batch_target_tensor[i, :target_length] = torch.from_numpy(batch_target[i])
            mask[i, :, :target_length] = torch.ones(self.num_classes, target_length)

        return batch_input_tensor, batch_target_tensor, mask, batch


if __name__ == "__main__":
    pass
