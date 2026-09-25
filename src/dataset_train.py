from network import Network
from tensor import Tensor


network = Network(2, [4, 2])

dataset = [
    (1.0, 2.0, 3.0),
    (2.0, 3.0, 5.0),
    (3.0, 4.0, 7.0),
    (4.0, 5.0, 9.0),
]

learning_rate = 0.01


for epoch in range(1000):

    total_loss = 0.0

    for a, b, target_value in dataset:

        x1 = Tensor(a)
        x2 = Tensor(b)
        target = Tensor(target_value)

        outputs = network.forward([x1, x2])

        prediction = outputs[0]

        loss = (prediction - target) ** 2

        for parameter in network.parameters():
            parameter.grad = 0.0

        loss.backward()

        for parameter in network.parameters():
            parameter.data -= learning_rate * parameter.grad

        total_loss += loss.data

    if epoch % 100 == 0:
        average_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch}: "
            f"average loss={average_loss:.6f}"
        )


print("\nTesting:")

tests = [
    (5.0, 6.0),
    (6.0, 7.0),
]

for a, b in tests:

    x1 = Tensor(a)
    x2 = Tensor(b)

    outputs = network.forward([x1, x2])

    print(
        f"{a} + {b} = "
        f"{outputs[0].data:.4f}"
    )