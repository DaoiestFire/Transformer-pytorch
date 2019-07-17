"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from model.transformer import TransformerEncoder, TransformerDecoder, TransformerModel
from model.modules.sparse_activations import LogSparsemax
from model.modules import Embeddings, CopyGenerator
from utils.logging import logger
from utils.parse import ArgumentParser


class Cast(nn.Module):
    """
    Basic layer that casts its input to a specific data type. The same tensor
    is returned if the data type is already correct.
    """

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)


def build_embeddings(opt, tokenizer, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        tokenizer(Tokenizer): EasyTokenizer.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    # 自适应 encoder / decoder 的 embed
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size
    # 这里需要修改，修改为使用tokenizer提取pad idx
    word_padding_idx = tokenizer.PAD_idx

    num_word_embeddings = len(tokenizer.vocab)

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        word_vocab_size=num_word_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs)
    return emb


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    tokenizer = checkpoint['tokenizer']

    model = build_base_model(model_opt, opt.use_gpu, tokenizer, checkpoint,
                             opt.gpu_id)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return tokenizer, model, model_opt


def build_base_model(model_opt, gpu, tokenizer, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        gpu (bool): whether to use gpu.
        tokenizer: tokenizer used to build embedding layer, if opt.share_tokenizer = true
                   tokenizer is a EasyTokenizer instance else is a dice contain {'src','tgt'}.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # Build source embeddings.
    if model_opt.share_tokenizer:
        src_emb = build_embeddings(model_opt, tokenizer)
    else:
        src_emb = build_embeddings(model_opt, tokenizer['src'])
    # Build encoder.
    encoder = TransformerEncoder.from_opt(model_opt, src_emb)

    # Build target embeddings.
    if model_opt.share_tokenizer:
        tgt_emb = build_embeddings(model_opt, tokenizer, for_encoder=False)
    else:
        tgt_emb = build_embeddings(model_opt,
                                   tokenizer['tgt'],
                                   for_encoder=False)
    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        if not model_opt.share_tokenizer:
            # src/tgt vocab should be the same if `-share_vocab` is specified.
            assert tokenizer['src'].vocab == tokenizer['tgt'].vocab, \
                "preprocess with -share_vocab if you use share_embeddings"
        tgt_emb.word_lut.weight = src_emb.word_lut.weight
    # Build decoder.
    decoder = TransformerDecoder.from_opt(model_opt, src_emb)

    # Build TransformerModel(= encoder + decoder).
    model = TransformerModel(encoder, decoder)

    # Build Generator.
    # copy attention 是另一个论文提出的技术
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(
                model_opt.dec_dim_size,
                len(tokenizer.vocab)
                if model_opt.share_tokenizer else len(tokenizer['tgt'].vocab)),
            Cast(torch.float32), gen_func)
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        if model_opt.share_tokenizer:
            vocab_size = len(tokenizer.vocab)
            pad_idx = tokenizer.PAD_idx
        else:
            vocab_size = len(tokenizer['tgt'].vocab)
            pad_idx = tokenizer['tgt'].PAD_idx
        generator = CopyGenerator(model_opt.dec_dim_size, vocab_size, pad_idx)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {
            fix_key(k): v
            for k, v in checkpoint['model'].items()
        }
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        # 判断如何初始化参数
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        # 用xavier初始化
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        # 使用预训练词嵌入层参数
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)
    # 生成部分
    model.generator = generator
    if gpu and (gpu_id is not None):
        logger.info("use %d GPU." % gpu_id)
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        logger.info("use default GPU.")
        device = torch.device("cuda")
    elif not gpu:
        logger.info("use CPU.")
        device = torch.device("cpu")
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        model.half()

    return model


def build_model(model_opt, opt, tokenizer, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt,
                             opt.use_gpu,
                             tokenizer,
                             checkpoint,
                             gpu_id=opt.gpu_id)
    logger.info(model)
    return model
