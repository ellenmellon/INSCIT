import glob
import logging
import os
import pickle

import torch

from utils import dist_utils


logger = logging.getLogger()

class ReaderDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        paths = glob.glob(os.path.join(data_dir, '*'))

        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
            logger.info(f"Data paths: {paths}")

        assert paths, "No Data files found."
        data = []
        for path in paths:
            with open(path, "rb") as f:
                data.extend(pickle.load(f))

        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(data)}")

        for d in data:
            d.to_tensor()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)