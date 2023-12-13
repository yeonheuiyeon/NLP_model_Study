# coding=utf-8
# Copyright 2021 The LG AI Research EXAONE Lab
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LG AI Research EXAONE Lab"""
import sys
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    QuestionAnsweringModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_exaone import ExaoneConfig
from torch.nn.utils import skip_init
import math
import numpy as np
from typing import List, Optional, Tuple, Union


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_func, 
        flash_attn_kvpacked_func, 
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func, 
    )
    from flash_attn.bert_padding import unpad_input, pad_input
    flash_attn_v2_installed = True
    print('>>>> Flash Attention installed')
except ImportError:
    flash_attn_v2_installed = False
    print('>>>> Flash Attention Not-installed Normal attention use only')
    print('If you want to use Flash Attnetion, Please install Flash Attention: `pip install flash-attn --no-build-isolation` and change the config file attention name global to flash')

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "exaone"
_CONFIG_FOR_DOC = "ExaoneConfig"

EXAONE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "exaone",
]


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, _, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, :, None, :].expand(batch, slen, 2, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, 2, num_key_value_heads * n_rep, head_dim)

#copy from gpt-j
def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    dtype= x.dtype # recover the original dtype when return
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    half_embedding_dim = dim // 2
    fraction = 2.0 * torch.arange(0, half_embedding_dim) / dim
    inv_freq = 1 / (10000 ** fraction)
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp).to(dtype), torch.cos(sinusoid_inp).to(dtype)

#[JUNWON] pre-defined version of pos embedding
def fixed_pos_embedding_factory(base, dim, max_seq_length, scaling_factor=1, expand_length=None):
    half_embedding_dim = dim // 2
    fraction = 2.0 * torch.arange(0, half_embedding_dim) / dim
    if expand_length:
        base = base * ((scaling_factor * expand_length / max_seq_length) - (scaling_factor - 1)) ** (dim / (dim - 2))
        inv_freq = 1 / (base ** fraction)
        pos = torch.arange(expand_length, dtype=torch.float32)
    else:
        inv_freq = 1 / (base ** fraction)
        pos = torch.arange(max_seq_length, dtype=torch.float32)
        pos = pos / scaling_factor
    sinusoid_inp = torch.einsum("i , j -> i j", pos, inv_freq)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 1)  # repeat all elements into the 2nd dimension    diff with lingvo code
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :], sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    first_half, second_half = torch.tensor_split(x, 2 , dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return torch.cat([first_part, second_part], dim=-1) # [SUNKYOUNG] FIX CONCAT TO CAT

def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def _concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 3 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


def _make_3block_relative_position_ids(block_len: int) -> torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    causal_mask = relative_position_ids <= 0
    causal_mask = causal_mask[None, None, :, :]
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    attention_mask = torch.logical_and(local_attention_mask, locality_mask)
    return torch.logical_and(attention_mask, causal_mask)


def _get_local_attention_mask(attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # [batch_size, 1, num_block, block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(1).to(attention_mask.device)

def load_tf_weights_in_exaone(model, config, exaone_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(exaone_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    name_raw = set()
    name_map = dict()
    all_name = []
    for name, shape in init_vars:
        if "global_step" not in name and "Adam" not in name:
            if "Adafactor" in name:
                continue
        if "transformer/decoder/blocks/" in name:
            name = name.replace("transformer/decoder/blocks/", "")
            name = name.split('/')[0]
            name_raw.add(name)

    name_raw = sorted(name_raw)

    for idx, name in enumerate(name_raw):
        name_map[name] = f'h{idx}'

    data_dict = {}

    for name, shape in init_vars:
        if "global_step" not in name:
            #Adafactor optimizer
            if "Adafactor" in name:
                continue
            #Adam optimizer
            if "Adam" in name or "beta1_power" in name or "beta2_power" in name:
                continue
            array = tf.train.load_variable(tf_path, name)
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            if config.use_kerple:
                name  = name.replace("dec_self_kerple_attention", "dec_self_attention")
            elif config.use_alibi_pos:
                name  = name.replace("dec_self_alibi_attention", "dec_self_attention")
            elif config.use_rpe:
                name = name.replace("dec_self_relative_attention", "dec_self_attention")

            name = name.replace("transformer/", "")
            name = name.replace("dec_emb/w/embedding/var", "wte")
            if config.use_emb_norm:
                name = name.replace("dec_emb/ln/w/scale/var", "ln_emb/g")
            name = name.replace("dec_pos_emb/w/embedding/var", "wpe")

            if "decoder/blocks/iter_" in name:
                name = name.replace("decoder/blocks/", "")
                name = name.replace(name.split('/')[0], name_map[name.split('/')[0]])

            if config.use_kerple:
                name = name.replace("blocks_body/layer_000/dec_self_attention/kerple_bias/kerple_bias_a/var", "attn/attention/kerple_bias_a")
                name = name.replace("blocks_body/layer_000/dec_self_attention/kerple_bias/kerple_bias_p/var", "attn/attention/kerple_bias_p")

            name = name.replace("blocks_body/layer_000/dec_self_attention/w/wk/var", "attn/attention/k_proj/w")
            name = name.replace("blocks_body/layer_000/dec_self_attention/w/wq/var", "attn/attention/q_proj/w")
            name = name.replace("blocks_body/layer_000/dec_self_attention/w/wv/var", "attn/attention/v_proj/w")
            name = name.replace("blocks_body/layer_000/dec_self_attention/w/wo/var", "attn/attention/out_proj/w")
            if config.use_rpe:
                name = name.replace("blocks_body/layer_000/dec_self_attention/wrb/wrb/var", "attn/attention/relative_attention_bias")
            name = name.replace("blocks_body/layer_000/ln/w/scale/var", "ln_1/g")
            name = name.replace("blocks_body/layer_001/ln/w/scale/var", "ln_2/g")

            if config.use_gated:
                name = name.replace("blocks_body/layer_001/dense_relu_dense/w/wi_0/var", "mlp/c_fc_0/w")
                name = name.replace("blocks_body/layer_001/dense_relu_dense/w/wi_1/var", "mlp/c_fc_1/w")
                name = name.replace("blocks_body/layer_001/dense_relu_dense/w/wo/var", "mlp/c_proj/w")
            else:
                name = name.replace("blocks_body/layer_001/dense_relu_dense/w/wi/var", "mlp/c_fc/w")
                name = name.replace("blocks_body/layer_001/dense_relu_dense/w/wo/var", "mlp/c_proj/w")
            name = name.replace("decoder/final_layer_norm/w/scale/var", "ln_f/g")
            data_dict[name] = array
            print(name, shape)

    for (name, array) in data_dict.items():
        name = name.split("/")
        pointer = model.transformer
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "wpe":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "relative_attention_bias":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if name[-1] == "w" and name[-2] in ["out_proj", "k_proj", "q_proj", "v_proj", "c_proj", "c_fc", 'c_fc_0', 'c_fc_1']:
            array = array.transpose()

        if name[-1] in ['kerple_bias_a', 'kerple_bias_p']:
            array = array[:, np.newaxis, np.newaxis]
        if name[-1] in ['relative_attention_bias']:
            array = array.transpose()

        if name == ["wte"]:
            # if vocab is padded, then trim off the padding embeddings
            array = array[: config.vocab_size]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)

    # init the final linear layer using word embeddings
    embs = model.transformer.wte.weight
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)
    lin.weight = embs
    model.set_output_embeddings(lin)
    return model

#[HYUNJIK] Lingvo porting
def rmsnorm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)

#[HYUNJIK] Lingvo porting
def rmsnorm_no_scale_func(hidden_states, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

class LingvoNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, ln_no_scale=False):
        super(LingvoNorm, self).__init__()
        self.eps = eps
        self.d = dim
        self.ln_no_scale = ln_no_scale
        if self.ln_no_scale is False:
            weight = torch.nn.Parameter(torch.ones(dim))
            self.register_parameter("weight", weight)
    
    def forward(self, hidden_states):
        if self.ln_no_scale is False:
            return rmsnorm_func(hidden_states, self.weight, self.eps)
        else:
            return rmsnorm_no_scale_func(hidden_states, self.eps)

class LingvoSoftmax(torch.nn.Module):
    def __init__(self):
        super(LingvoSoftmax, self).__init__()
        zero = torch.tensor([0.], dtype=torch.float32)
        self.register_buffer('zero', zero, persistent=False)

    def _LogSoftmax(self, x):
        return x - self._ReduceLogsumexp(x)

    def _ReduceLogsumexp(self, x):
        extra_logit = self.zero
        max_logit, _ = torch.max(x, dim=-1, keepdim=True)
        max_logit = torch.maximum(max_logit, extra_logit)
        exp_x = torch.exp(x-max_logit)
        sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True) +  torch.exp(extra_logit - max_logit)
        return torch.log(sum_exp_x) + max_logit

    def forward(self, x):
        return torch.exp(self._LogSoftmax(x))

class ExaoneAttentionMixin:
    """
    A few attention related utilities for attention modules in GPT Lingvo, to be used as a mixin.
    """

    def _split_heads(self, tensor, num_heads, attn_head_size, rotary):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        if rotary:
            return tensor
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, causal_mask, attn_dropout, attention_mask=None, head_mask=None, alibi_bias=None, rpe=None):
        #[HYUNJIK] Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))  # [batch, head, seq_len, seq_len]
       
        if self.use_alibi_pos:
            attn_weights += alibi_bias #[batch, 1, head, seq_len]

        if self.use_kerple:
            self.kerple_bias_p.data = self.kerple_bias_p.data.clamp(min=1e-2)
            self.kerple_bias_a.data = self.kerple_bias_a.data.clamp(min=1e-2)
            query_length, key_length = query.size(-2), key.size(-2)
            kerple_bias = -self.kerple_bias_p*torch.log(1+self.kerple_bias_a*self.cached_matrix) # log kernel

            if query_length != key_length:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
                assert (
                query_length == 1
                ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
                if type(kerple_bias) != float:
                    # seq_len_k - 1 points to the last token index in the current inference batch.
                    kerple_bias = kerple_bias[:, key_length - 1, :].view(kerple_bias.shape[0], 1, kerple_bias.shape[2])
            attn_weights+= kerple_bias

        # mask_value = torch.finfo(attn_weights.dtype).min
        # # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        # # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        # attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        attn_weights = torch.where(causal_mask, attn_weights, self.fmin.to(query.dtype))
        #[HYUNJIK] 175B attention logit cap

        if self.use_logit_cap:
            cap = self.cap.to(query.device)
            attn_weights = cap * torch.tanh(attn_weights / cap)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        #print("attn_weights", attn_weights[0][:10])
        if self.lingvosoftmax:
            attn_weights = self.lingvosoftmax(attn_weights)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query.dtype)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)         #[HYUNJIK]
        return attn_output, attn_weights

class ExaoneLocalAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1
        self.head_dim = self.embed_dim // self.num_heads
        self.inner_dim = self.num_heads * self.head_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.lingvosoftmax = LingvoSoftmax()
        self.dropout = nn.Dropout(float(config.resid_dropout))
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        is_padded_inputs=False,
        use_cache=False,
        output_attentions=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.num_heads, self.head_dim)

        def unshape(states):
            """reshape"""
            return states.contiguous().view(batch_size, -1, self.inner_dim)

        # get query/key/value states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q_proj(hidden_states))
        key_states = shape(self.k_proj(hidden_states))
        value_states = shape(self.v_proj(hidden_states))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Compute scores
        scores = torch.einsum(
            "...qhd,...khd->...hqk", query_states, key_states
        )  # (batch_size, num_block, n_heads, block_len, 3 * block_len)
        scores += _get_local_attention_mask(attention_mask, self.block_len).transpose(1,2)
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = self.lingvosoftmax(scores.float()).type_as(scores)
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = self.dropout(attn_weights)

        # Mask heads if we want to
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.type(value_states.dtype)
        attn_output = unshape(torch.einsum("...hqk,...khd->...qhd", attn_weights, value_states))
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.out_proj(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class ExaoneSelfAttention(nn.Module, ExaoneAttentionMixin):
    def __init__(self, config):
        super().__init__()
        max_positions = config.max_position_embeddings
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        #[HYUNJIK] 175B attention logit cap
        self.use_logit_cap = config.use_logit_cap
        self.use_alibi_pos = config.use_alibi_pos
        self.use_rotary_pos = config.use_rotary_pos
        self.use_kerple = config.use_kerple
        self.use_rpe = config.use_rpe
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.use_extra_logit = config.use_extra_logit
        self.rotary_base = config.rotary_base
        bias=None

        if self.use_rotary_pos:
            self.head_size = self.embed_dim // self.num_heads
            self.rotary_ndims = int(self.head_size * config.rotary_pct)
            scaling_factor = config.scaling_factor
            expand_length = config.rotary_expand_length
            
            if config.rotary_type=="linear":
                sin_emb, cos_emb = fixed_pos_embedding_factory(base=self.rotary_base, dim=self.rotary_ndims, max_seq_length=expand_length, scaling_factor=scaling_factor)
            elif config.rotary_type=="ntk":
                sin_emb, cos_emb = fixed_pos_embedding_factory(base=self.rotary_base, dim=self.rotary_ndims, max_seq_length=max_positions, scaling_factor=scaling_factor, expand_length=expand_length)
            else:
                sin_emb, cos_emb = fixed_pos_embedding_factory(base=self.rotary_base, dim=self.rotary_ndims, max_seq_length=max_positions)
            if config.rotary_type:
                max_positions = expand_length
            
            self.register_buffer('sin_emb', sin_emb, persistent=False)
            self.register_buffer('cos_emb', cos_emb, persistent=False)

        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions)
        self.register_buffer("bias", bias, persistent=False)

        self.cap = torch.FloatTensor([50.0])
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        if self.use_extra_logit:
            self.lingvosoftmax = LingvoSoftmax()
        else:
            self.lingvosoftmax = None

        # _dtype = torch.get_default_dtype()
        _dtype = torch.float32
        fmin = torch.tensor(torch.finfo(_dtype).min, dtype=_dtype)
        self.register_buffer('fmin', fmin)

        if self.use_alibi_pos:
            alibi = self.build_alibi_tensor(max_positions, self.num_heads)
            self.register_buffer("alibi", alibi)

        if self.use_rpe:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

        if self.use_kerple:
            def get_parameter(scale, init_method):
                if init_method == 'ones':
                    return nn.Parameter(torch.ones(self.num_heads)[:,None,None]*scale)
                elif init_method == 'uniform':
                    return nn.Parameter(torch.rand(self.num_heads)[:,None,None]*scale)
            self.kerple_bias_p = get_parameter(2, 'uniform')
            self.kerple_bias_a = get_parameter(1, 'uniform')
            kerple_diff = torch.tril(torch.arange(max_positions).view(max_positions, 1).repeat(1, max_positions)
                    + torch.arange(0, -max_positions, -1))
            self.register_buffer("kerple_diff", kerple_diff)
            self.cached_matrix = None
            self.cached_seq_len = None
            self.is_past = False

    def build_alibi_tensor(self, max_positions, n_head, dtype=torch.float32):
        """
        Alibi tensor is not causal as the original paper mentions, it relies on a translation invariance of softmax for
        quick implementation: with l being a tensor, and a fixed value `softmax(l+a) = softmax(l) Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        Args:
        Returns tensor shaped (n_head, 1, max_positions)
            max_positions: (`int`, *required*):
                max sequence length
            n_head: (`int`, *required*):
                number of heads
            dtype: (`torch.dtype`, *optional*, default=`torch.bfloat16`):
                dtype of the output tensor
        """

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2])

        slopes = torch.Tensor(get_slopes(n_head)).unsqueeze(1).unsqueeze(1)
        arange_tensor = torch.arange(max_positions).unsqueeze(0).unsqueeze(0)
        alibi = slopes * arange_tensor.expand(n_head, -1, -1).unsqueeze(0)
        alibi = alibi.to(dtype)
        return alibi

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        is_padded_inputs=False,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query, self.num_heads, self.head_dim, self.use_rotary_pos)
        key = self._split_heads(key, self.num_heads, self.head_dim, self.use_rotary_pos)
        value = self._split_heads(value, self.num_heads, self.head_dim, False)

        if self.use_rotary_pos:                                          # [HYUNJIK] rotary embedding apply
            seq_len = key.shape[1]
            offset = 0
            if layer_past is not None:
                offset = layer_past[0].shape[-2]
                seq_len += offset

            k_rot = key[:, :, :, : self.rotary_ndims]
            k_pass = key[:, :, :, self.rotary_ndims :]

            q_rot = query[:, :, :, : self.rotary_ndims]
            q_pass = query[:, :, :, self.rotary_ndims :]

            # sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            # k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            # q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)
            sincos = (self.sin_emb[:seq_len].to(query.dtype), self.cos_emb[:seq_len].to(query.dtype))
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)

            key = key.permute(0, 2, 1, 3)
            query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        batch_size, query_length, key_length = query.size(0), query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        if self.use_kerple:
            if self.cached_seq_len != key_length:
                diff = torch.tril(
                    torch.arange(key_length, device=query.device).view(key_length, 1).repeat(1, key_length)
                    + torch.arange(0, -key_length, -1, device=query.device)
                )
                diff = diff.to(query.dtype)
                self.cached_seq_len = key_length
                self.cached_matrix = diff

        if self.use_alibi_pos:
            alibi_bias = self.alibi[:, :, :, :key_length].repeat([batch_size, 1, 1, 1])
            attn_output, attn_weights = self._attn(
            query, key, value, causal_mask, self.attn_dropout, attention_mask, head_mask, alibi_bias)
        elif self.use_rpe:
            position_bias = self.compute_bias(key_length, key_length, device=query.device)
            position_bias = position_bias[:, :,  key_length - query_length:key_length, :]
            attn_output, attn_weights = self._attn(
            query, key, value, causal_mask, self.attn_dropout, attention_mask, head_mask, rpe=position_bias)
        else:
            attn_output, attn_weights = self._attn(
            query, key, value, causal_mask, self.attn_dropout, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class ExaoneFlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_positions = config.max_position_embeddings
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        #[HYUNJIK] 175B attention logit cap
        self.use_logit_cap = config.use_logit_cap
        self.use_alibi_pos = config.use_alibi_pos
        self.use_rotary_pos = config.use_rotary_pos
        self.use_kerple = config.use_kerple
        self.use_rpe = config.use_rpe
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.use_extra_logit = config.use_extra_logit
        self.rotary_base = config.rotary_base
        bias=None

        if self.use_rotary_pos:
            self.head_size = self.embed_dim // self.num_heads
            self.rotary_ndims = int(self.head_size * config.rotary_pct)
            scaling_factor = config.scaling_factor
            expand_length = config.rotary_expand_length
            #print(f"{config.rotary_type}, scaling_factor:{config.scaling_factor}, expand_length:{expand_length}")
            
            if config.rotary_type=="linear":
                sin_emb, cos_emb = fixed_pos_embedding_factory(base=self.rotary_base, dim=self.rotary_ndims, max_seq_length=expand_length, scaling_factor=scaling_factor)
            elif config.rotary_type=="ntk":
                sin_emb, cos_emb = fixed_pos_embedding_factory(base=self.rotary_base, dim=self.rotary_ndims, max_seq_length=max_positions, scaling_factor=scaling_factor, expand_length=expand_length)
            else:
                sin_emb, cos_emb = fixed_pos_embedding_factory(base=self.rotary_base, dim=self.rotary_ndims, max_seq_length=max_positions)
            if config.rotary_type:
                max_positions = expand_length
            
            self.register_buffer('sin_emb', sin_emb, persistent=False)
            self.register_buffer('cos_emb', cos_emb, persistent=False)

        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions)
        self.register_buffer("bias", bias, persistent=False)

        self.cap = torch.FloatTensor([50.0])
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        if self.use_extra_logit:
            self.lingvosoftmax = LingvoSoftmax()
        else:
            self.lingvosoftmax = None

        # _dtype = torch.get_default_dtype()
        _dtype = torch.float32
        fmin = torch.tensor(torch.finfo(_dtype).min, dtype=_dtype)
        self.register_buffer('fmin', fmin)

        if self.use_alibi_pos:
            alibi = self.build_alibi_tensor(max_positions, self.num_heads)
            self.register_buffer("alibi", alibi)

        if self.use_rpe:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

        if self.use_kerple:
            def get_parameter(scale, init_method):
                if init_method == 'ones':
                    return nn.Parameter(torch.ones(self.num_heads)[:,None,None]*scale)
                elif init_method == 'uniform':
                    return nn.Parameter(torch.rand(self.num_heads)[:,None,None]*scale)
            self.kerple_bias_p = get_parameter(2, 'uniform')
            self.kerple_bias_a = get_parameter(1, 'uniform')
            kerple_diff = torch.tril(torch.arange(max_positions).view(max_positions, 1).repeat(1, max_positions)
                    + torch.arange(0, -max_positions, -1))
            self.register_buffer("kerple_diff", kerple_diff)
            self.cached_matrix = None
            self.cached_seq_len = None
            self.is_past = False

    def build_alibi_tensor(self, max_positions, n_head, dtype=torch.float32):
        """
        Alibi tensor is not causal as the original paper mentions, it relies on a translation invariance of softmax for
        quick implementation: with l being a tensor, and a fixed value `softmax(l+a) = softmax(l) Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        Args:
        Returns tensor shaped (n_head, 1, max_positions)
            max_positions: (`int`, *required*):
                max sequence length
            n_head: (`int`, *required*):
                number of heads
            dtype: (`torch.dtype`, *optional*, default=`torch.bfloat16`):
                dtype of the output tensor
        """

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2])

        slopes = torch.Tensor(get_slopes(n_head)).unsqueeze(1).unsqueeze(1)
        arange_tensor = torch.arange(max_positions).unsqueeze(0).unsqueeze(0)
        alibi = slopes * arange_tensor.expand(n_head, -1, -1).unsqueeze(0)
        alibi = alibi.to(dtype)
        return alibi

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        is_padded_inputs=False,
        use_cache=False,
        output_attentions=False,
    ):
        bsz, q_len, h_size = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim)
        key = key.view(bsz, q_len, self.num_heads, self.head_dim)
        value = value.view(bsz, q_len, self.num_heads, self.head_dim)

        if self.use_rotary_pos:                                          # [HYUNJIK] rotary embedding apply
            seq_len = key.shape[1]
            offset = 0
            if layer_past is not None:
                offset = layer_past[0].shape[-2]
                seq_len += offset

            q_rot = query[:, :, :, : self.rotary_ndims]
            q_pass = query[:, :, :, self.rotary_ndims :]
            k_rot = key[:, :, :, :self.rotary_ndims]
            k_pass = key[:, :, :, self.rotary_ndims :]

            # sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            # k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            # q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)
            if q_rot.dtype!=self.sin_emb.dtype:
                self.sin_emb = self.sin_emb.to(q_rot.dtype)
                self.cos_emb = self.cos_emb.to(q_rot.dtype)
            sincos = (self.sin_emb[:seq_len], self.cos_emb[:seq_len])
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)

            query = torch.cat([q_rot, q_pass], dim=-1)
            key = torch.cat([k_rot, k_pass], dim=-1)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        kv = torch.stack([key, value], 2)
        #num_key_value_groups = 1 # num_heads // num_key_value_heads = 128 // 128
        #kv = repeat_kv(kv, num_key_value_groups) # for multi query attention
        if is_padded_inputs:
            assert attention_mask is not None
            unpadded_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, attention_mask)
            unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query, attention_mask[:, -query.size(1):])
            attn_outputs = flash_attn_varlen_kvpacked_func(
                unpadded_q, unpadded_kv, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k,
                dropout_p=0.0, softmax_scale=1.0, 
                causal=(not layer_past), return_attn_probs=output_attentions
            )

            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            
            attn_output = pad_input(
                attn_output, indices_q, bsz, q_len
            ).reshape(bsz, q_len, h_size)
            
            attn_weights = attn_outputs[2] if output_attentions else None

        else:
            attn_outputs = flash_attn_kvpacked_func(
                    query, kv, dropout_p=0.0, softmax_scale=1.0, causal=(not layer_past), return_attn_probs=output_attentions)
            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            attn_output = attn_output.reshape(bsz, q_len, h_size)
            attn_weights = attn_outputs[2] if output_attentions else None

        attn_output = self.out_proj(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class ExaoneAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]
        if 'local' in self.attention_type:
            self.attention = ExaoneLocalAttention(config)
        elif 'flash' in self.attention_type:
            self.attention = ExaoneFlashAttention(config)
        else:
            self.attention = ExaoneSelfAttention(config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        is_padded_inputs=False,
        use_cache=False,
        output_attentions=False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            is_padded_inputs=is_padded_inputs,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

#[HYUNJIK] Ptuning layer
class PtuningMLP(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.functional.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class PromptEncoderMLP(nn.Module):
    def __init__(self, template, hidden_size, device, dropout=0):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.dropout = dropout
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool()
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(device)
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.dropout),
                                      nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(input_embeds).squeeze()
        return output_embeds

class PromptEncoderLSTM(nn.Module):
    def __init__(self, template, hidden_size, device, lstm_dropout=0):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.lstm_dropout = lstm_dropout
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool()
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size)
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size // 2,
                                num_layers=2,
                                dropout=self.lstm_dropout,
                                bidirectional=True,
                                batch_first=True)

        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds

class PromptDistributed(nn.Module):
    def __init__(self, template, hidden_size, device, dropout=0):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.dropout = dropout
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool()
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size)
        self.mlp_head = PtuningMLP(self.hidden_size, self.dropout)

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(input_embeds).squeeze()
        return output_embeds


class ExaoneMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.c_proj = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ExaoneGatedMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.c_fc_0 = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.c_fc_1 = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.c_proj = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_act = self.act(self.c_fc_0(hidden_states))
        hidden_linear = self.c_fc_1(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class ExaoneBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = LingvoNorm(dim = hidden_size, eps=config.layer_norm_epsilon, ln_no_scale = config.ln_no_scale)
        self.attn = ExaoneAttention(config, layer_id)
        self.ln_2 = LingvoNorm(dim = hidden_size, eps=config.layer_norm_epsilon, ln_no_scale = config.ln_no_scale)
        if config.use_gated:
            self.mlp = ExaoneGatedMLP(inner_dim, config)
        else:
            self.mlp = ExaoneMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        is_padded_inputs=False,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        #print("after_ln1", hidden_states[0][:10])
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            is_padded_inputs=is_padded_inputs,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual
        #print("after_attn", hidden_states[0][:10])

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class ExaonePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ExaoneConfig
    load_tf_weights = load_tf_weights_in_exaone
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ExaoneBlock"]
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LingvoNorm):
            if self.config.ln_no_scale is False:
                module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ExaoneModel):
            module.gradient_checkpointing = value

EXAONE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (:class:`~transformers.ExaoneConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

EXAONE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0][0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.

            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.num_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.

            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GPT Lingvo Model transformer outputting raw hidden-states without any specific head on top.",
    EXAONE_START_DOCSTRING,
)
class ExaoneModel(ExaonePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.use_absolute_pos = config.use_absolute_pos
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        if self.config.use_emb_norm:
            self.ln_emb = LingvoNorm(dim = self.embed_dim, eps=config.layer_norm_epsilon, ln_no_scale= config.ln_no_scale)
        if self.use_absolute_pos:
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(float(config.embed_dropout))
        self.h = nn.ModuleList([ExaoneBlock(config, layer_id=i) for i in range(config.num_layers)])
        self.ln_f = LingvoNorm(dim=self.embed_dim, eps=config.layer_norm_epsilon, ln_no_scale=config.ln_no_scale)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(EXAONE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_padded_inputs: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        # Attention mask.
        if attention_mask is not None:
            if self.config.attention_layers[0]!="flash":                   
                if batch_size <= 0:
                    raise ValueError("batch_size has to be defined and > 0")
                attention_mask = attention_mask.view(batch_size, -1)
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x num_heads x N x N
            # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            if self.config.use_emb_norm:
                inputs_embeds = self.ln_emb(inputs_embeds)
        #print("inputs_embeds", inputs_embeds[0][:10])
        if self.use_absolute_pos:
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        #print("after_emb", hidden_states[0][:10])

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    is_padded_inputs,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    is_padded_inputs=is_padded_inputs,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """
    The GPT Lingvo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    EXAONE_START_DOCSTRING,
)
class ExaoneForCausalLM(ExaonePreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ExaoneModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            # for P-Tuning by Janghoon.Han
            inputs_embeds = None
        #past를 사용하면 이전 임베딩 정보가 들어갔기 때문에 따로 임베딩 처리 안함
        else:
            inputs_embeds = kwargs.get("inputs_embeds", None)
            if inputs_embeds is not None:
                input_ids = None

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "inputs_embeds": inputs_embeds,
            "is_padded_inputs": ((attention_mask is not None) and (not attention_mask.all().item()))
        }

    @add_start_docstrings_to_model_forward(EXAONE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_padded_inputs: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        is_padded_inputs = ((attention_mask is not None) and (not attention_mask.all().item()))

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_padded_inputs=is_padded_inputs,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = hidden_states * (self.config.hidden_size ** (-0.5)) #[HYUNJIK]
        #print(hidden_states[:10])       
        lm_logits = self.lm_head(hidden_states)
        #print(torch.argmax(lm_logits,dim=-1))
        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


@add_start_docstrings(
    """
    The Exaone Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.ExaoneForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """,
    EXAONE_START_DOCSTRING,
)
class ExaoneForSequenceClassification(ExaonePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.attention\.masked_bias", r"mask", r"alibi", r"lm_head.weight"]
    _keys_to_ignore_on_save = [r"mask", r"alibi", r"masked_bias"]
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ExaoneModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EXAONE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

@add_start_docstrings(
    """
    The GPT-Lingvo Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    EXAONE_START_DOCSTRING,
)
class ExaoneForQuestionAnswering(ExaonePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.attention\.masked_bias", r"h\.\d+\.attn\.attention\.bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ExaoneModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits
        )
