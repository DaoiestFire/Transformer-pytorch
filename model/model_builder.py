"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from model.transformer import TransformerEncoder, TransformerDecoder, TransformerModel
# import onmt.inputters as inputters
import onmt.modules
# from onmt.encoders import str2enc

# from onmt.decoders import str2dec

from model.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser


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
        fix_word_vecs=fix_word_vecs
    )
    return emb

def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


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
    if opt.share_tokenizer:
        src_emb = build_embeddings(model_opt, tokenizer, src_field)
    else:
        src_emb = build_embeddings(model_opt, tokenizer['src'], src_field)
    # Build encoder.
    encoder = TransformerEncoder.from_opt(model_opt, src_emb)

    # Build target embeddings.
    if opt.share_tokenizer:
        tgt_emb = build_embeddings(model_opt, tokenizer, for_encoder=False)
    else:
        tgt_emb = build_embeddings(model_opt, tokenizer['tgt'], for_encoder=False)
    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        if not opt.share_tokenizer:
            # src/tgt vocab should be the same if `-share_vocab` is specified.
            assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
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
            gen_func = model.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_dim_size,
                      len(tokenizer.vocal) if opt.share_tokenizer else len(tokenizer['tgt'].vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
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

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
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

    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        model.half()00000000000000

    return model


def build_model(model_opt, opt, tokenizer, checkpoint):
    logger.info('Building model...')
    gpu_id = None
    if len(opt.gpus) != 0:
        # use the first GPU in given list.
        gpu_id = opt.gpus[0]
    model = build_base_model(model_opt, opt.use_gpu, tokenizer, checkpoint, gpu_id=gpu_id)
    logger.info(model)
    return model
