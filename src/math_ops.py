from tensor import Tensor


def dot_product(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    result = Tensor(0.0)

    for a, b in zip(vector_a, vector_b):
        result = result + (a * b)

    return result


if __name__ == "__main__":

    vector_a = [
        Tensor(1.0),
        Tensor(2.0),
        Tensor(3.0)
    ]

    vector_b = [
        Tensor(4.0),
        Tensor(5.0),
        Tensor(6.0)
    ]

    result = dot_product(vector_a, vector_b)

    print("Vector A:")
    print([x.data for x in vector_a])

    print("\nVector B:")
    print([x.data for x in vector_b])

    print("\nDot product:")
    print(result)

    result.backward()

    print("\nGradients for Vector A:")
    print([x.grad for x in vector_a])

    print("\nGradients for Vector B:")
    print([x.grad for x in vector_b])