#!/usr/bin/env python
"""Train models."""
import os
import signal
import torch

import opts
import onmt.utils.distributed

from onmt.utils.logging import logger
from onmt.train_single import main as single_main
from utils.parse import ArgumentParser

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, MultipleDatasetIterator
from model.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)


    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        # 将模型参数导入到CPU
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
    else:
        checkpoint = None
        model_opt = opt
        
    # Build Tokenizaer


################################
    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    # 输出参数个数
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    # 检查模型保存路径，如果不存在则创建
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)

    train_iterables = []
    if len(opt.data_ids) > 1:
        for train_id in opt.data_ids:
            shard_base = "train_" + train_id
            iterable = build_dataset_iter(shard_base, fields, opt, multi=True)
            train_iterables.append(iterable)
        train_iter = MultipleDatasetIterator(train_iterables, device_id, opt)
    else:
        train_iter = build_dataset_iter("train", fields, opt)

    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()

########################
    if opt.use_gpu:
        if opt.gpus == []:
            pass
        else:
            pass
    else:
        # use cpu
        pass

        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def _get_parser():
    parser = ArgumentParser(description='train.py')
    # Construct config
    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
