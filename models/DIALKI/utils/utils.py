import os
import logging

from .dist_utils import is_local_master

logger = logging.getLogger()

def print_section_bar(string, width=80):
    if is_local_master():
        logger.info('='*width)
        half_width = (width - len(string)) // 2
        logger.info(' '*half_width + string)
        logger.info('='*width)


def convert_to_at_k(lst):
    # we evaluate @k so if the k-1 span matches, then k, k+1, k+2
    # will also be regarded as match.
    at_k = [-1] * len(lst)
    for i in range(len(lst)):
        if lst[i] == -1:
            break
        # sum until itself
        at_k[i] = min(sum(lst[:i+1]), 1)
    return at_k


def softlink(target, link_name):
    temp_link = link_name + '.new'
    try:
        os.remove(temp_link)
    except OSError:
        pass
    os.symlink(target, temp_link)
    os.rename(temp_link, link_name)


def print_args(args):
    print_section_bar('*'*16 + 'CONFIGURATION' + '*'*16)
    for key, val in sorted(vars(args).items()):
        logger.info(f'{key:<30} -->   {val}')
