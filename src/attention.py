from tensor import Tensor
from math_ops import dot_product
from softmax import softmax
import math


def scaled_dot_product_attention(query, keys, values):
    if len(keys) != len(values):
        raise ValueError("Keys and values must have the same length.")

    if len(keys) == 0:
        raise ValueError("Keys and values cannot be empty.")

    scores = []

    scale = math.sqrt(len(query))

    for key in keys:
        score = dot_product(query, key)
        score = score / scale
        scores.append(score)

    weights = softmax(scores)

    output = []

    for dimension in range(len(values[0])):
        result = Tensor(0.0)

        for weight, value in zip(weights, values):
            result = result + weight * value[dimension]

        output.append(result)

    return output, weights
