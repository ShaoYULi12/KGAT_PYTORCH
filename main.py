import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import random
import logging
import argparse
from time import time
from log_helper import *
import torch
import numpy as np
from dataloader import DataLoaderKGAT
from parse import *

def train_KGAT(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

        # load data
    data = DataLoaderKGAT(args, logging)

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None







if __name__ == '__main__':
    args = parse_kgat_args()
    train_KGAT(args)