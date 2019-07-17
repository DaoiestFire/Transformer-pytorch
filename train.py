#!/usr/bin/env python
"""Train models."""
import os
import signal
import torch
import opts
import codecs
import random
from tokenization import Tokenizer
from utils.logging import init_logger, logger
from utils.parse import ArgumentParser
from sklearn.model_selection import train_test_split
from model.model_builder import build_model
from model.optimizers import Optimizer
# from trainer import build_trainer
from model.model_saver import build_model_saver
import numpy as np
from tqdm import tqdm


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def preProcessData(opt, tokenizer):
    """
    pre-process the training data. 
    """
    src = []
    target = []

    # process src data
    if opt.share_tokenizer:
        t = tokenizer
    else:
        t = tokenizer['src']
    with codecs.open(opt.src_corpus, 'r', encoding='utf-8') as f:
        max_len = 0
        for line in tqdm(f, desc="Source Data"):
            idx_sqe = t.convert_tokens_to_ids(t.tokenize(line, add_eos=True))
            max_len = len(idx_sqe) if len(idx_sqe) > max_len else max_len
            src.append(idx_sqe)
        for i in tqdm(range(len(src)), desc="Padding Source Data"):
            src[i] = t.pad(src[i], max_len)

    # process target data
    if opt.share_tokenizer:
        t = tokenizer
    else:
        t = tokenizer['tgt']
    with codecs.open(opt.tgt_corpus, 'r', encoding='utf-8') as f:
        max_len = 0
        for line in tqdm(f, desc="Target Data"):
            idx_sqe = t.convert_tokens_to_ids(t.tokenize(line, add_eos=True))
            max_len = len(idx_sqe) if len(idx_sqe) > max_len else max_len
            target.append(idx_sqe)
        for i in tqdm(range(len(target)), desc="Padding Source Data"):
            target[i] = t.pad(target[i], max_len)

    src = np.array(src)
    target = np.array(target)
    src_train, src_test, target_train, target_test = train_test_split(
        src, target, test_size=0.1)
    # save
    np.save(os.path.join(opt.data, "src_train.npy"), src_train)
    np.save(os.path.join(opt.data, "src_test.npy"), src_test)
    np.save(os.path.join(opt.data, "target_train.npy"), target_train)
    np.save(os.path.join(opt.data, "target_test.npy"), target_test)

    return src_train, src_test, target_train, target_test


def main(opt):
    # ArgumentParser.validate_train_opts(opt)
    # ArgumentParser.update_model_opts(opt)
    # ArgumentParser.validate_model_opts(opt)
    set_random_seed(opt.seed, opt.use_gpu)
    init_logger(opt.log_file)
    # assert len(opt.accum_count) == len(opt.accum_steps), \
    #     'Number of accum_count values must match number of accum_steps'
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
    # Load Tokenizaer
    if opt.share_tokenizer and os.path.exists(
            os.path.join(opt.tnz_load_from, 'tnz.pkl')):
        tokenizer = Tokenizer.load(
            file_path=os.path.join(opt.tnz_load_from, 'tnz.pkl'))
    elif not opt.share_tokenizer and os.path.exists(
            os.path.join(opt.tnz_load_from, 'src_tnz.pkl')) and os.path.exists(
                os.path.join(opt.tnz_load_from, 'tgt_tnz.pkl')):
        tokenizer = {}
        tokenizer['src'] = Tokenizer.load(
            file_path=os.path.join(opt.tnz_load_from, 'src_tnz.pkl'))
        tokenizer['tgt'] = Tokenizer.load(
            file_path=os.path.join(opt.tnz_load_from, 'tgt_tnz.pkl'))
    else:
        if not os.path.exists(opt.tnz_load_from):
            os.makedirs(opt.tnz_load_from)
        # Build Tokenizer
        if opt.share_tokenizer:
            tokenizer = Tokenizer(split_type=opt.split_type,
                                  do_lower_case=opt.do_lower_case,
                                  bert_vocab_file=opt.bert_vocab_file,
                                  max_len=opt.max_len,
                                  never_split=("<PAD>", "<UNK>", "<EOS>",
                                               "<S>"))
            tokenizer.build_vocab(corpus_path=opt.corpus)
            # save tokenizer
            tokenizer.save(
                file_path=os.path.join(opt.tnz_load_from, 'tnz.pkl'))
        else:
            tokenizer = {}
            tokenizer['src'] = Tokenizer(split_type=opt.src_split_type,
                                         do_lower_case=opt.do_lower_case,
                                         bert_vocab_file=opt.bert_vocab_file,
                                         max_len=opt.max_len,
                                         never_split=("<PAD>", "<UNK>",
                                                      "<EOS>", "<S>"))
            tokenizer['tgt'] = Tokenizer(split_type=opt.src_split_type,
                                         do_lower_case=opt.do_lower_case,
                                         bert_vocab_file=opt.bert_vocab_file,
                                         max_len=opt.max_len,
                                         never_split=("<PAD>", "<UNK>",
                                                      "<EOS>", "<S>"))
            tokenizer['src'].build_vocab(corpus_path=opt.src_corpus)
            tokenizer['tgt'].build_vocab(corpus_path=opt.tgt_corpus)
            # save tokenizer
            tokenizer['src'].save(
                file_path=os.path.join(opt.tnz_load_from, 'src_tnz.pkl'))
            tokenizer['tgt'].save(
                file_path=os.path.join(opt.tnz_load_from, 'tgt_tnz.pkl'))
    # preprocess data
    if not os.path.exists(os.path.join(opt.data, "src_train.npy")):
        preProcessData(opt, tokenizer)
    else:
        src_train = np.load(os.path.join(opt.data, "src_train.npy"))
        src_test = np.load(os.path.join(opt.data, "src_test.npy"))
        target_train = np.load(os.path.join(opt.data, "target_train.npy"))
        target_test = np.load(os.path.join(opt.data, "target_test.npy"))

    # Build model.
    model = build_model(model_opt, opt, tokenizer, checkpoint)
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
    model_saver = build_model_saver(model_opt, opt, model, tokenizer, optim)

############################################################
# test
    src = torch.from_numpy(np.load(r'data\all\src_test.npy')[1:2]).long()
    tgt = torch.from_numpy(np.load(r'data\all\target_test.npy')[1:2]).long()
    src = src.transpose(1,0)
    tgt = tgt.transpose(1,0)
    print('src.size:',src.size())
    dec_out, attns = model(src, tgt, len(src))
    print('dec_out',dec_out.size())
    print('attns',attns)



# test end

###########################################################
# trainer = build_trainer(opt,
#                         device_id,
#                         model,
#                         fields,
#                         optim,
#                         model_saver=model_saver)

# train_iterables = []
# if len(opt.data_ids) > 1:
#     for train_id in opt.data_ids:
#         shard_base = "train_" + train_id
#         iterable = build_dataset_iter(shard_base, fields, opt, multi=True)
#         train_iterables.append(iterable)
#     train_iter = MultipleDatasetIterator(train_iterables, device_id, opt)
# else:
#     train_iter = build_dataset_iter("train", fields, opt)

# valid_iter = build_dataset_iter("valid", fields, opt, is_train=False)

# if len(opt.gpu_ranks):
#     logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
# else:
#     logger.info('Starting training on CPU, could be very slow')
# train_steps = opt.train_steps
# if opt.single_pass and train_steps > 0:
#     logger.warning("Option single_pass is enabled, ignoring train_steps.")
#     train_steps = 0
# trainer.train(train_iter,
#               train_steps,
#               save_checkpoint_steps=opt.save_checkpoint_steps,
#               valid_iter=valid_iter,
#               valid_steps=opt.valid_steps)

# if opt.tensorboard:
#     trainer.report_manager.tensorboard_writer.close()


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
    # print(opt.enc_dim_size)

    main(opt)
