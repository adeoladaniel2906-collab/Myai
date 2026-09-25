from tensor import Tensor


def relu(x):
    if x.data > 0:
        result = Tensor(x.data, (x,), "ReLU")

        def _backward():
            x.grad += result.grad

        result._backward = _backward

        return result

    else:
        result = Tensor(0.0, (x,), "ReLU")

        def _backward():
            x.grad += 0.0

        result._backward = _backward

        return result


if __name__ == "__main__":
    positive = Tensor(5.0)
    negative = Tensor(-3.0)

    positive_result = relu(positive)
    negative_result = relu(negative)

    positive_result.backward()
    negative_result.backward()

    print("Positive:")
    print(positive_result)
    print("Input gradient:", positive.grad)

    print("\nNegative:")
    print(negative_result)
    print("Input gradient:", negative.grad)