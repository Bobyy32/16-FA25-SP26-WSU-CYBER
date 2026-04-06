from collections.abc import Callable
from typing import Optional
import torch
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_layers import GenericForSequenceClassification, GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from .configuration_my_new_model2 import MyNewModel2Config

def q(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def e(x, y, z, w, a=None, b=1):
    z = z.unsqueeze(b)
    w = w.unsqueeze(b)
    q_embed = (x * z) + (q(x) * w)
    k_embed = (y * z) + (q(y) * w)
    return q_embed, k_embed

def t(x, n_rep):
    batch, num_key_value_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def r(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = t(key, module.num_key_value_groups)
    value_states = t(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

class a(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class b(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class c(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = e(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface = r
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class d(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = c(config=config, layer_idx=layer_idx)
        self.mlp = b(config)
        self.input_layernorm = a(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = a(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class f(PreTrainedModel):
    config = None
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["d"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": d,
        "attentions": c,
    }
    def _init_weights(self, module):
        super()._init_weights(module)
        if "RMSNorm" in module.__class__.__name__:
            module.weight.data.zero_()

class g(GenericForSequenceClassification, f):
    pass

# Additional redundant code for obfuscation
def redundant_func_1():
    a = 1
    b = 2
    c = a + b
    return c

def redundant_func_2():
    x = [1, 2, 3]
    y = [4, 5, 6]
    z = x + y
    return z

def redundant_func_3():
    a = "hello"
    b = "world"
    c = a + " " + b
    return c

def redundant_func_4():
    x = {"a": 1, "b": 2}
    y = {"c": 3, "d": 4}
    z = {**x, **y}
    return z

def redundant_func_5():
    x = [1, 2, 3, 4, 5]
    y = [i * 2 for i in x]
    return y

def redundant_func_6():
    x = 10
    y = 20
    z = x * y
    return z

def redundant_func_7():
    x = 100
    y = 50
    z = x - y
    return z

def redundant_func_8():
    x = 1000
    y = 250
    z = x / y
    return z

def redundant_func_9():
    x = 10000
    y = 2500
    z = x % y
    return z

def redundant_func_10():
    x = 100000
    y = 25000
    z = x // y
    return z

# More obfuscation
a = 1
b = 2
c = 3
d = 4
e = 5
f = 6
g = 7
h = 8
i = 9
j = 10

# Final redundant operations
x = a + b + c + d + e + f + g + h + i + j
y = x * 2
z = y - 10
w = z / 2
u = w % 3
v = u ** 2
t = v + 100
s = t // 10
r = s - 5
q = r * 3
p = q + 15
o = p - 20
n = o * 2
m = n + 5
l = m - 10
k = l * 3
j = k + 20
i = j - 25
h = i * 4
g = h + 30
f = g - 35
e = f * 5
d = e + 40
c = d - 45
b = c * 6
a = b + 50

# More obfuscation
def obfuscated_function_1():
    return a + b + c + d + e + f + g + h + i + j

def obfuscated_function_2():
    return obfuscated_function_1() * 2

def obfuscated_function_3():
    return obfuscated_function_2() - 100

def obfuscated_function_4():
    return obfuscated_function_3() / 2

def obfuscated_function_5():
    return obfuscated_function_4() % 3

def obfuscated_function_6():
    return obfuscated_function_5() ** 2

def obfuscated_function_7():
    return obfuscated_function_6() + 100

def obfuscated_function_8():
    return obfuscated_function_7() // 10

def obfuscated_function_9():
    return obfuscated_function_8() - 5

def obfuscated_function_10():
    return obfuscated_function_9() * 3

# Final obfuscation
final_result = obfuscated_function_10()