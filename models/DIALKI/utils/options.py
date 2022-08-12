import argparse
import logging
import os
import random
import socket

import numpy as np
import torch

from models import loss
from utils import dist_utils

logger = logging.getLogger()


def add_data_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--do_lower_case',
        action='store_true',
        help=('Whether to lower case the input text. True for uncased models, '
              'False for cased models.'))
    parser.add_argument(
        '--max_num_answers',
        default=2,
        type=int,
        help='Max amount of answer spans to marginalize per single passage')
    parser.add_argument(
        '--passages_per_question',
        type=int,
        default=2,
        help='Total amount of positive and negative passages per question')
    parser.add_argument(
        '--passages_per_question_predict',
        type=int,
        default=50,
        help=('Total amount of positive and negative passages per question for '
              'evaluation'))
    parser.add_argument(
        '--max_answer_length',
        default=5,
        type=int,
        help=('The maximum length of an answer (in spans) that can be '
              'generated. This is needed because the start and end predictions '
              'are not conditioned on one another.'))
    parser.add_argument(
        '--special_attention',
        action='store_true',
        help=('using special attention to limit the range a question or a '
              'passage can attend to'))
    parser.add_argument(
        '--passage_attend_history',
        action='store_true',
        help=('tokens in a passage can attend to history question. '
             '(information leak?)'))
    parser.add_argument(
        '--data_name',
        required=True,
        type=str,
        choices=['inscit'],
        help='The name of the dataset.')


def add_encoder_params(parser: argparse.ArgumentParser):
    """Common parameters to initialize an encoder-based model."""
    
    parser.add_argument(
        '--pretrained_model_cfg',
        default=None,
        type=str,
        help='Path of the pre-trained model.')
    parser.add_argument(
        '--checkpoint_file',
        default=None,
        type=str,
        help='Trained checkpoint file to initialize the model.')
    parser.add_argument(
        '--projection_dim',
        default=0,
        type=int,
        help='Extra linear layer on top of standard bert/roberta encoder.')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=512,
        help='Max length of the encoder input sequence.')
    parser.add_argument(
        '--dropout',
        default=0.1,
        type=float,
        help='')
    parser.add_argument(
        '--use_coordinator',
        action='store_true',
        help=('Whether to use a coordinator to contexualize passages with '
              'other passage vector'))
    parser.add_argument(
        '--coordinator_layers',
        default=1,
        type=int,
        help='Number of hidden layers for the passage coordinator')
    parser.add_argument(
        '--coordinator_heads',
        default=3,
        type=int,
        help='Number of attention heads for the passage coordinator')
    parser.add_argument(
        '--num_token_types',
        default=20,
        type=int,
        help='Number of possiblen token types')
    parser.add_argument(
        '--ignore_token_type',
        action='store_true',
        help='Whether to ignore token types or not')
    parser.add_argument(
        '--compute_da_loss',
        action='store_true',
        help='Whether to jointly train dialog act prediction or not')
    parser.add_argument(
        '--decision_function',
        type=int,
        default=0,
        help='Which decision function to use for calculating loss')
    parser.add_argument(
        '--hist_loss_weight',
        type=float,
        default=1.0,
        help='weight of history loss')
    parser.add_argument(
        '--user2agent_loss_weight',
        default=0,
        type=float,
        help=('predict a history agent span based on the previous user '
              'question if > 0'))
    parser.add_argument(
        '--span_marker',
        action='store_true',
        help='mark spans used in history')
    parser.add_argument(
        '--skip_mark_last_user',
        action='store_true',
        help=('skip add mark embeddings of the last user turn to span '
              'embeddings'))
    parser.add_argument(
        '--marker_after_steps',
        default=0,
        type=int,
        help='not using marker in the begining of the training process')
    parser.add_argument(
        '--use_z_attn',
        action='store_true',
        help='')


def add_f_div_regularization_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--adv_epsilon',
        default=1e-6,
        type=float,
        help='for adv training')
    parser.add_argument(
        '--adv_step_size',
        default=1e-5,
        type=float,
        help='for adv training')
    parser.add_argument(
        '--adv_noise_var',
        default=1e-5,
        type=float,
        help='for adv training')
    parser.add_argument(
        '--adv_norm_p',
        default='inf',
        type=str,
        help='for adv training')
    parser.add_argument(
        '--adv_norm_level',
        default=0,
        type=int,
        help='for adv training')
    parser.add_argument(
        '--adv_k',
        default=1,
        type=int,
        help='for adv training')
    parser.add_argument(
        '--adv_calc_logits_keys',
        nargs='+',
        default=['start', 'end', 'relevance'],
        help='for adv training')
    parser.add_argument(
        '--adv_loss_weight',
        default=0.0,
        type=float,
        help='for adv training')
    parser.add_argument(
        '--adv_loss_type',
        default='hl',
        choices=loss.LOSS.keys(),
        type=str,
        help='for adv training')


def add_training_params(parser: argparse.ArgumentParser):
    """Common parameters for training."""
    parser.add_argument(
        '--train_file',
        default=None,
        type=str,
        help='File pattern for the train set.')
    parser.add_argument(
        '--dev_file',
        default=None,
        type=str,
        help='File pattern for the dev set.')
    parser.add_argument(
        '--batch_size',
        default=2,
        type=int,
        help='Amount of questions per batch.')
    parser.add_argument(
        '--dev_batch_size',
        type=int,
        default=4,
        help='amount of questions per batch for dev set validation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed for initialization and dataset shuffling.')
    parser.add_argument(
        '--adam_eps',
        default=1e-8,
        type=float,
        help='Epsilon for Adam optimizer.')
    parser.add_argument(
        '--adam_betas',
        default='(0.9, 0.999)',
        type=str,
        help='Betas for Adam optimizer.')
    parser.add_argument(
        '--max_grad_norm',
        default=1.0,
        type=float,
        help='Max gradient norm.')
    parser.add_argument(
        '--log_batch_step',
        default=100,
        type=int,
        help='Number of steps to log during training.')
    parser.add_argument(
        '--train_rolling_loss_step',
        default=100,
        type=int,
        help='Number of steps of interval to save traning loss.')
    parser.add_argument(
        '--weight_decay',
        default=0.0,
        type=float,
        help='Weight decay for optimizer.')
    parser.add_argument(
        '--learning_rate',
        default=1e-5,
        type=float,
        help='Learning rate.')
    parser.add_argument(
        '--warmup_steps',
        default=100,
        type=int,
        help='Linear warmup over warmup_steps.')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of update steps to accumulate before updating parameters.')
    parser.add_argument(
        '--num_train_epochs',
        default=3.0,
        type=float,
        help='Total number of training epochs to perform.')
    parser.add_argument(
        '--auto_resume',
        action='store_true',
        help='Auto resume from latest checkpoint')
    parser.add_argument(
        '--save_checkpoint_every_minutes',
        type=int,
        default=15,
        help='Save a checkpoint every x minutes')
    parser.add_argument(
        '--topk_em',
        type=int,
        default=2,
        help='Topk checkpoints according to EM metrics.')
    parser.add_argument(
        '--topk_f1',
        type=int,
        default=2,
        help='Topk checkpoints according to F1 metrics.')
    parser.add_argument(
        '--best_metric',
        type=str,
        choices=['em', 'f1'],
        help='Take the best model based on EM or F1 scores.')
    parser.add_argument(
        '--eval_step',
        default=2000,
        type=int,
        help='Batch steps to run validation and save checkpoint.')
    parser.add_argument(
        '--eval_top_docs',
        nargs='+',
        type=int,
        help=('Top retrival passages thresholds to analyze prediction results '
              'for'))
    parser.add_argument(
        '--checkpoint_filename_prefix',
        type=str,
        default='dialki',
        help='Checkpoint filename prefix.')
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
        help='Output directory for checkpoints.')
    parser.add_argument(
        '--inference_only',
        action='store_true',
        help='Inference only.')
    parser.add_argument(
        '--prediction_results_file',
        type=str,
        help='Path to a file to write prediction results to')


def add_cuda_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='The parameter for distributed training.')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to use 16-bit float precision instead of 32-bit.')
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help=('For fp16: Apex AMP optimization level selected.'
              'See details at https://nvidia.github.io/apex/amp.html.'))


def get_encoder_checkpoint_params_names():
    return [
        'do_lower_case',
        'pretrained_model_cfg',
        'projection_dim',
        'max_seq_len',
    ]


def get_encoder_params_state(args):
    """
    Selects the param values to be saved in a checkpoint, so that a trained
    model faile can be used for downstream tasks without the need to specify
    these parameter again.

    Return: Dict of params to memorize in a checkpoint.
    """
    params_to_save = get_encoder_checkpoint_params_names()

    r = {}
    for param in params_to_save:
        r[param] = getattr(args, param)
    return r


def set_encoder_params_from_state(state, args):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [
        (param, state[param])
        for param in params_to_save
        if param in state and state[param]
    ]
    for param, value in override_params:
        if param == "pretrained_model_cfg":
            continue
        if hasattr(args, param):
            if dist_utils.is_local_master():
                logger.warning(
                    f'Overriding args parameter value from checkpoint state. '
                    f'{param = }, {value = }')
        setattr(args, param, value)
    return args


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_args_gpu(args):
    """
    Setup arguments CUDA, GPU & distributed training.
    """

    world_size = os.environ.get('WORLD_SIZE')
    world_size = int(world_size) if world_size else 1
    args.distributed_world_size = world_size
    local_rank = args.local_rank
  
    if local_rank == -1:
        # Single-node multi-gpu (or cpu) mode.
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        n_gpu = args.n_gpu = torch.cuda.device_count()
    else: 
        # Distributed mode.
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        # set up the master's ip address so this child process can coordinate
        torch.distributed.init_process_group(
            backend='nccl',
            rank=args.local_rank,
            world_size=world_size)
        n_gpu = args.n_gpu = 1
    args.device = device

    if dist_utils.is_local_master():
        logger.info(
            f'Initialized host {socket.gethostname()}'
            f'{local_rank = } {device = } {n_gpu = } {world_size = }'
            f'16-bits training: {args.fp16}')
