from tensor import Tensor
import math

def cross_entropy(probabilities, target_index):
    if len(probabilities) == 0:
        raise ValueError("Probabilities cannot be empty.")

    if target_index < 0 or target_index >= len(probabilities):
        raise ValueError("Target index is out of range.")

    target_probability = probabilities[target_index]

    if target_probability.data <= 0:
        raise ValueError("Target probability must be greater than zero.")

    return -target_probability.log()


if __name__ == "__main__":
    probabilities = [
        Tensor(0.1),
        Tensor(0.2),
        Tensor(0.7)
    ]

    loss = cross_entropy(probabilities, 2)
    loss.backward()

    print("Loss:", loss.data)
    print("Target probability:", probabilities[2].data)
    print("Target gradient:", probabilities[2].grad)
