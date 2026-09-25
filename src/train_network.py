from network import Network
from tensor import Tensor


# Create the network:
# 2 inputs → 3 neurons → 2 outputs
network = Network(2, [3, 2])

# Training data
x1 = Tensor(2.0)
x2 = Tensor(3.0)

# We want the first output to learn 10
target = Tensor(10.0)

learning_rate = 0.01


for step in range(200):

    # Forward pass
    outputs = network.forward([x1, x2])

    prediction = outputs[0]

    # Loss
    loss = (prediction - target) ** 2

    # Reset all gradients
    for parameter in network.parameters():
        parameter.grad = 0.0

    # Backpropagation
    loss.backward()

    # Gradient descent
    for parameter in network.parameters():
        parameter.data -= learning_rate * parameter.grad

    if step % 20 == 0:
        print(
            f"Step {step}: "
            f"prediction={prediction.data:.4f}, "
            f"loss={loss.data:.4f}"
        )


# Test after training
final_outputs = network.forward([x1, x2])

print("\nFinal outputs:")

for i, output in enumerate(final_outputs):
    print(f"Output {i}: {output.data:.4f}")

print("\nTarget:", target.data)

print("\nParameters:", len(network.parameters()))