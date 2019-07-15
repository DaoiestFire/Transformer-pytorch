# coding=utf-8
"""Tokenization classes."""
from __future__ import absolute_import, division, print_function, unicode_literals
import unicodedata
import os
import re
import glob
import codecs
import pickle
from collections import Counter, OrderedDict
from tqdm import tqdm
import utils.logging

logger = utils.logging.init_logger(log_file='./log.txt')


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = OrderedDict()
    index = 0
    with codecs.open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(
        r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''',
        r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("<UNK>", "<EOS>", '<S>', "<SEP>", "<PAD>")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # 将每个中文字符左右两端加入空格
        text = self._tokenize_chinese_chars(text)
        # 使用空格分词
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                # 去除重音字符
                token = self._run_strip_accents(token)
            # 将标点符号拆分为单独的token
            split_tokens.extend(self._run_split_on_punc(token))
        # 使用空格分词 list：[[token1], [token2], ...]
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 在每个中文字符左右两端加上空格，用于后续分词操作。
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="<UNK>", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 当token字符长度大于阈值时，定义为unk_token
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                # extend 为扩充数组
                output_tokens.extend(sub_tokens)
        return output_tokens


class Tokenizer(object):
    """
    Tokenizer:
        支持basic，word_piece，spacy（多种语言https://spacy.io/usage/models#languages）等多种分词方式，构建词汇表。
    """

    def __init__(self,
                 split_type='basic',
                 do_lower_case=True,
                 bert_vocab_file=None,
                 max_len=None,
                 never_split=("<UNK>", "<EOS>", '<S>', "<SEP>", "<PAD>")):
        """Constructs a Tokenizer.

        Args:
            type: 指定分词方式，默认使用basic方式分词。
                'basic' ——  使用标点空格分词，不限语言。
                'word_piece' ——  使用词片分词，使用BERT预训练模型提供的vocabulary作为基词汇表。
                '<spacy code>' —— 使用spacy后端分词，支持多种语言，需要提供具体语言的spacy code，例如：'en'为英文,'zh'为中文...具体参见：https://spacy.io/usage/models#languages
            do_lower_case: Whether to lower case the input
            bert_vocab_file: Path to a one-wordpiece-per-line vocabulary file.
                        Only has an effect when type='word_piece'.
            max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying model's
                         sequence length.
            never_split: List of tokens which will never be split during tokenization.
        """
        self.split_type = split_type
        if split_type == 'basic':
            self.basic_tokenizer = BasicTokenizer(do_lower_case=True,
                                                  never_split=never_split)
        elif split_type == 'word_piece':
            assert bert_vocab_file and os.path.exists(bert_vocab_file), \
                "To use 'word_piece' split type, you need to specify the correct BERT pre-trained vocabulary file path."
            self.basic_tokenizer = BasicTokenizer(do_lower_case=True,
                                                  never_split=never_split)
            # 加载bert预训练vocab
            bert_vocab = load_vocab(bert_vocab_file)
            self.wordpiece_tokenizer = WordpieceTokenizer(
                bert_vocab, unk_token="<UNK>", max_input_chars_per_word=100)
        else:
            # 这里记得加入 assert 判断是否是spacy支持的字段
            assert split_type in [
                'en', 'zh'
            ], "the split_type %s cannot be resolved." % (split_type)
            import ftfy
            import spacy
            self.sapcy_tokenizer = spacy.blank(split_type)
            self.fix_text = ftfy.fix_text
        self.vocab = None
        self.id2tkn = None
        self.never_split = never_split
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text, add_eos=False, add_double_eos=False):
        '''Segment the text into list of tokens.
            Input: 
                text —— single text of string.
            return:
                split_tokens —— list of tokens.
        '''
        split_tokens = []
        if self.split_type == 'basic':
            for token in self.basic_tokenizer.tokenize(text):
                split_tokens.append(token)
        elif self.split_type == 'word_piece':
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            doc = self.sapcy_tokenizer(text_standardize(self.fix_text(text)))
            split_tokens = [str(t) for t in doc]
        if add_double_eos:
            return ['<S>'] + split_tokens + ['<S>']
        elif add_eos:
            return split_tokens + ['<EOS>']
        else:
            return split_tokens

    def tokenize_batch(self,
                       text_list,
                       add_eos=False,
                       add_double_eos=False,
                       batch_size=512,
                       n_threads=3):
        '''Segment the batch of text into list of tokens.
            Input: 
                text_list —— list of string. etc. ['this is text1', 'this is text2', ...]
            return:
                split_tokens —— list of tokens. [['this', 'is', 'text1'], ['this', 'is', 'text2'], ...]
        '''
        texts_token = []
        if self.split_type == 'basic' or self.split_type == 'word_piece':
            for text in text_list:
                texts_token.append(self.tokenize(text, add_eos,
                                                 add_double_eos))
        else:
            if add_double_eos:
                for doc in self.sapcy_tokenizer.pipe(text_list,
                                                     batch_size=batch_size,
                                                     n_threads=n_threads):
                    texts_token.append(['<S>'] + [str(t)
                                                  for t in doc] + ['<S>'])
            elif add_eos:
                for doc in self.sapcy_tokenizer.pipe(text_list,
                                                     batch_size=batch_size,
                                                     n_threads=n_threads):
                    texts_token.append([str(t) for t in doc] + ['<EOS>'])
            else:
                for doc in self.sapcy_tokenizer.pipe(text_list,
                                                     batch_size=batch_size,
                                                     n_threads=n_threads):
                    texts_token.append([str(t) for t in doc])

        return texts_token

    def build_vocab(self,
                    corpus_path,
                    min_freq=0,
                    max_size=None,
                    file_suffix='txt'):
        logger.info("building vocab...")
        file_list = []
        if os.path.isdir(corpus_path):
            logger.info("corpus path is a directory")
            # 获取目录下所有'.txt'文件路径
            file_list = glob.glob(
                os.path.join(corpus_path, '*.%s' % file_suffix))
        elif os.path.isfile(corpus_path):
            logger.info("corpus path is a normal file")
            file_list.append(corpus_path)
        else:
            logger.warn("corpus path is not exist.")
            return None
        counter = Counter()
        for file_path in file_list:
            with codecs.open(file_path, 'r', encoding='utf-8') as f:
                buffer = []
                for line in tqdm(f, desc=file_path.split("\\")[-1]):
                    buffer.append(line)
                    if len(buffer) == 2048:
                        symbols = self.tokenize_batch(buffer)
                        counter.update(
                            [item for sublist in symbols for item in sublist])
                        buffer = []

        logger.info('building vocab with min_freq={}, max_size={}'.format(
            min_freq, max_size))
        self.vocab = []
        self.tkn2idx = OrderedDict()
        # 加入一些指定的特殊字符
        for tkn in self.never_split:
            self.add_special(tkn)
        # counter底层是用堆维护的，用于top N问题
        for tkn, cnt in counter.most_common(max_size):
            if cnt < min_freq:
                break
            self.add_token(tkn)

        logger.info('final vocab size {} from {} unique tokens'.format(
            len(self.vocab), len(counter)))

    def add_special(self, tkn):
        '''向词汇表中认为加入特殊字符，例如<UNK>，<PAD>等，需要以<>包裹'''
        if tkn not in self.vocab:
            self.vocab.append(tkn)
            self.tkn2idx[tkn] = len(self.vocab) - 1
            setattr(self, '{}_idx'.format(tkn.strip('<>')), self.tkn2idx[tkn])

    def add_token(self, tkn):
        '''向词汇表中加入新的词汇'''
        if tkn not in self.vocab:
            self.vocab.append(tkn)
            self.tkn2idx[tkn] = len(self.vocab) - 1

    def save(self, file_path='tokenizer.pkl'):
        with open(file_path, 'wb') as f:
            # serialize and save object
            pickle.dump(self, f)

    @classmethod
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            # read file and build object
            return pickle.load(f)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        if not self.tkn2idx:
            print('You must build vocab first.')
            return None
        ids = []
        for token in tokens:
            if token in self.tkn2idx:
                ids.append(self.tkn2idx[token])
            else:
                ids.append(self.tkn2idx["<UNK>"])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this model ({} > {}). Running this"
                " sequence through will result in indexing errors".format(
                    len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        if not self.vocab:
            print('You must build vocab first.')
            return None
        tokens = []
        for i in ids:
            if i >=0 and i < len(self.vocab):
                tokens.append(self.vocab[i])
            else:
                tokens.append("<UNK>")
        return tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
