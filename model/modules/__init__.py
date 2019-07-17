"""  Attention and normalization modules  """
# from modules.util_class import Elementwise
# from modules.gate import context_gate_factory, ContextGate
# from modules.global_attention import GlobalAttention
# from modules.conv_multi_step_attention import ConvMultiStepAttention
from model.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, CopyGeneratorLossCompute
# from modules.multi_headed_attn import MultiHeadedAttention
from model.modules.embeddings import Embeddings, PositionalEncoding
# from modules.weight_norm import WeightNormConv2d
# from modules.average_attn import AverageAttention

# __all__ = ["Elementwise", "context_gate_factory", "ContextGate",
#            "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
#            "CopyGeneratorLoss", "CopyGeneratorLossCompute",
#            "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
#            "WeightNormConv2d", "AverageAttention"]
# from model.modules import Embeddings, CopyGenerator