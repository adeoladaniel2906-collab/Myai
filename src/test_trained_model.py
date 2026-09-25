from transformer_model import TransformerModel
from language_head import LanguageHead
from checkpoint import load_checkpoint
from generate import generate

text = "hello world hello ai hello world hello ai"

model = TransformerModel(
    text,
    embedding_size=8,
    number_of_heads=2,
    hidden_size=16,
    max_length=32
)

language_head = LanguageHead(
    8,
    len(model.tokenizer.vocab)
)

parameters = (
    model.parameters()
    + language_head.parameters()
)

print("Model parameters:", len(parameters))

load_checkpoint(
    parameters,
    "fast_checkpoint_final.json"
)

print("Trained checkpoint loaded successfully.")

token_ids, hidden_states = model.forward("hello ")

print("Input tokens:", token_ids)
print("Hidden-state count:", len(hidden_states))

result = generate(
    model,
    language_head,
    "hello ",
    max_new_tokens=20
)

print("Generated text:")
print(result)
