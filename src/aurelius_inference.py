import torch
import torch.nn as nn

from gpu_model import GPUTransformerModel


class CachedAttentionBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__()

        self.attention = original_block.attention
        self.norm1 = original_block.norm1
        self.feed_forward = original_block.feed_forward
        self.norm2 = original_block.norm2

        self.head_dim = (
            self.attention.embed_dim // self.attention.num_heads
        )
        self.num_heads = self.attention.num_heads

    def project_qkv(self, x):
        weight = self.attention.in_proj_weight
        bias = self.attention.in_proj_bias

        q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = bias.chunk(3, dim=0)

        q = torch.nn.functional.linear(x, q_weight, q_bias)
        k = torch.nn.functional.linear(x, k_weight, k_bias)
        v = torch.nn.functional.linear(x, v_weight, v_bias)

        return q, k, v

    def reshape_heads(self, x):
        batch, seq, dim = x.shape

        x = x.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def forward(self, x, key_cache=None, value_cache=None):
        q, k, v = self.project_qkv(x)

        q = self.reshape_heads(q)
        k = self.reshape_heads(k)
        v = self.reshape_heads(v)

        if key_cache is not None:
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention_scores = attention_scores / (
            self.head_dim ** 0.5
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_output = torch.matmul(
            attention_weights,
            v
        )

        attention_output = attention_output.transpose(1, 2)

        batch, seq, _, _ = attention_output.shape

        attention_output = attention_output.reshape(
            batch,
            seq,
            self.num_heads * self.head_dim
        )

        attention_output = torch.nn.functional.linear(
            attention_output,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias
        )

        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(
            x + feed_forward_output
        )

        return x, k, v


class AureliusCachedInference:
    def __init__(self, model):
        self.model = model

        self.blocks = nn.ModuleList([
            CachedAttentionBlock(block)
            for block in model.blocks
        ])

        self.key_cache = [None] * len(self.blocks)
        self.value_cache = [None] * len(self.blocks)

    @torch.no_grad()
    def prefill(self, tokens):
        self.key_cache = [None] * len(self.blocks)
        self.value_cache = [None] * len(self.blocks)

        x = self.model.embedding(tokens)

        sequence_length = tokens.size(1)

        x = x + self.model.position_embedding[
            :, :sequence_length, :
        ]

        for i, block in enumerate(self.blocks):
            x, k, v = block(
                x,
                None,
                None
            )

            self.key_cache[i] = k
            self.value_cache[i] = v

        x = self.model.norm(x)

        return self.model.language_head(x[:, -1:, :])

    @torch.no_grad()
    def decode(self, token):
        sequence_position = self.key_cache[0].size(2)

        x = self.model.embedding(token)

        x = x + self.model.position_embedding[
            :, sequence_position:sequence_position + 1, :
        ]

        for i, block in enumerate(self.blocks):
            x, k, v = block(
                x,
                self.key_cache[i],
                self.value_cache[i]
            )

            self.key_cache[i] = k
            self.value_cache[i] = v

        x = self.model.norm(x)

        return self.model.language_head(x)


def load_cached_model(
    checkpoint_path,
    device="cuda"
):
    model = GPUTransformerModel()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(device=device, dtype=torch.float16)
    model.eval()

    cached_model = AureliusCachedInference(model)

    return cached_model
