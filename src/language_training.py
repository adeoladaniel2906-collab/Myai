from tokenizer import Tokenizer
from embedding import Embedding
from transformer_model import TransformerModel
from language_head import LanguageHead
from softmax import softmax
from loss import cross_entropy


text = "hello world hello ai"

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

token_ids, hidden_states = model.forward(text)

total_loss = None

for index in range(len(token_ids) - 1):
    hidden = hidden_states[index]

    logits = language_head.forward([[value for value in hidden]])[0]

    probabilities = softmax(logits)

    target_token = token_ids[index + 1]

    loss = cross_entropy(
        probabilities,
        target_token
    )

    if total_loss is None:
        total_loss = loss
    else:
        total_loss = total_loss + loss

average_loss = total_loss / (len(token_ids) - 1)

print("Token count:", len(token_ids))
print("Vocabulary size:", len(model.tokenizer.vocab))
print("Average loss:", average_loss.data)

average_loss.backward()

print("Backpropagation completed.")
print("First model parameter gradient:", model.parameters()[0].grad)
print("First language-head gradient:", language_head.parameters()[0].grad)
