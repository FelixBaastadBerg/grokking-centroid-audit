import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, vocab_size=None, bias=True, use_embedding=False, non_linearity=nn.ReLU):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activations = []
        self.activations_from_abs_input = None
        self.embedding = None
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, input_size // 2)
        self.layers = nn.ModuleList()
        self.non_linearity = non_linearity()
        self.uses_bias = bias
        self.alpha = 1
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(hidden_sizes)+1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias))
    
    def forward(self, x, keep_activations=False):
        if self.embedding:
            x = self.embedding(x)
        x = x.flatten(start_dim=1)
        if keep_activations:
            self.activations = []
        for i, layer in enumerate(self.layers):
            x = self.non_linearity(layer(x)) if i<len(self.layers) -1 else layer(x)
            if keep_activations and i<len(self.layers):
                self.activations.append(x)
        return x*self.alpha
    

class Transformer(nn.Module):
    """
    Encoder-style transformer for algorithmic tasks.

    NOTE (fix T1): the original version accepted `seq_len` and never used it, so
    it had NO positional information at all and was permutation-invariant over
    (a, b). That is harmless for commutative tasks (add, mul) but makes
    subtraction and division *unlearnable* -- the model cannot tell a-b from b-a.
    A learned positional embedding is now added; `use_positional=False` restores
    the old behaviour for ablation.
    """

    def __init__(self, d_model, num_heads, num_layers, vocab_size, seq_len,
                 norm_first=False, non_linearity=nn.ReLU, use_positional=True):
        super().__init__()
        self.seq_len = seq_len
        self.use_positional = use_positional
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model) if use_positional else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, activation=non_linearity(), batch_first=True, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x_emb = self.embedding(x)
        if self.pos_embedding is not None:
            pos = torch.arange(x.size(1), device=x.device)
            x_emb = x_emb + self.pos_embedding(pos).unsqueeze(0)
        out = self.transformer(x_emb)
        out = self.linear(out)
        return out
    
