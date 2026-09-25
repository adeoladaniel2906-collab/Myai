from tensor import Tensor
from nn import Neuron


# Create one neuron with two inputs
neuron = Neuron(2)

# Training data
x1 = Tensor(2.0)
x2 = Tensor(3.0)

# Target value
target = Tensor(10.0)

# Learning rate
learning_rate = 0.01


for step in range(100):

    # Forward pass
    output = neuron.forward([x1, x2])

    # Calculate loss automatically
    loss = (output - target) ** 2

    # Reset gradients
    for parameter in neuron.parameters():
        parameter.grad = 0.0

    # Backpropagation
    loss.backward()

    # Gradient descent
    for parameter in neuron.parameters():
        parameter.data -= learning_rate * parameter.grad

    if step % 10 == 0:
        print(
            f"Step {step}: "
            f"output={output.data:.4f}, "
            f"loss={loss.data:.4f}"
        )


# Final result
final_output = neuron.forward([x1, x2])

print("\nFinal result:")
print("Output:", final_output.data)
print("Target:", target.data)

print("\nLearned parameters:")

for i, weight in enumerate(neuron.weights):
    print(f"Weight {i}: {weight.data}")

print("Bias:", neuron.bias.data)