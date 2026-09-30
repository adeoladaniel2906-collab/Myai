
import torch
import torch.nn as nn


# ============================================================
# Aurelius 102M architecture
# ============================================================

AURELIUS_VOCAB_SIZE = 16_384
AURELIUS_D_MODEL = 512
AURELIUS_N_HEADS = 8
AURELIUS_N_LAYERS = 13
AURELIUS_D_FF = 5_376
AURELIUS_CONTEXT_LENGTH = 256

AURELIUS_PARAMETER_COUNT = 102_252_288


class GPUAttentionBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        sequence_length = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attention_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=False
        )

        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(x + feed_forward_output)

        return x


class GPUTransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size=AURELIUS_VOCAB_SIZE,
        d_model=AURELIUS_D_MODEL,
        n_heads=AURELIUS_N_HEADS,
        n_layers=AURELIUS_N_LAYERS,
        d_ff=AURELIUS_D_FF,
        context_length=AURELIUS_CONTEXT_LENGTH
    ):
        super().__init__()

        if vocab_size < 1:
            raise ValueError(
                "vocab_size must be greater than 0."
            )

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.context_length = context_length

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                context_length,
                d_model
            )
        )

        self.blocks = nn.ModuleList([
            GPUAttentionBlock(
                d_model,
                n_heads,
                d_ff
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.language_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, tokens):

        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape [batch, sequence_length]."
            )

        _, sequence_length = tokens.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds context length {self.context_length}."
            )

        x = self.embedding(tokens)

        x = x + self.position_embedding[
            :, :sequence_length, :
        ]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.language_head(x)

        return logits


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def verify_aurelius_parameters(model):

    actual = count_parameters(model)

    if actual != AURELIUS_PARAMETER_COUNT:
        raise RuntimeError(
            "Aurelius parameter count mismatch: "
            f"expected {AURELIUS_PARAMETER_COUNT:,}, "
            f"got {actual:,}"
        )

    return True
