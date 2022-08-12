import os
import glob
import torch
import logging
import collections

from .dist_utils import is_local_master

logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "amp_dict",
        "offset",
        "epoch",
        "global_step",
        "encoder_params",
    ],
)


def get_saved_checkpoints(args, file_prefix):
    cp_paths = []
    if args.output_dir:
        cp_paths = glob.glob(
            os.path.join(args.output_dir, file_prefix + "*"))

    if len(cp_paths) > 0:
        cp_paths = sorted(cp_paths, key=os.path.getctime)
    return cp_paths


def load_states_from_checkpoint(model_file):
    if is_local_master():
        logger.info(f"Reading saved model from {model_file}")
    state_dict = torch.load(model_file, map_location="cpu")
    if is_local_master():
        logger.info(f"model_state_dict keys {state_dict.keys()}")
    return CheckpointState(**state_dict)
