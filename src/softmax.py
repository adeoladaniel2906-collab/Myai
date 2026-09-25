from tensor import Tensor


def softmax(values):
    if len(values) == 0:
        raise ValueError("Softmax requires at least one value.")

    max_value = max(value.data for value in values)

    exponentials = []

    for value in values:
        shifted = value - max_value
        exponentials.append(shifted.exp())

    total = Tensor(0.0)

    for value in exponentials:
        total = total + value

    probabilities = []

    for value in exponentials:
        probabilities.append(value / total)

    return probabilities


if __name__ == "__main__":

    inputs = [
        Tensor(1.0),
        Tensor(2.0),
        Tensor(3.0)
    ]

    outputs = softmax(inputs)

    print("Inputs:")
    print([x.data for x in inputs])

    print("\nSoftmax:")
    print([x.data for x in outputs])

    print("\nSum:")
    total = sum(x.data for x in outputs)
    print(total)

    print("\nBackward test:")

    loss = outputs[0]

    for value in inputs:
        value.grad = 0.0

    loss.backward()

    print("Input gradients:")
    print([x.grad for x in inputs])
