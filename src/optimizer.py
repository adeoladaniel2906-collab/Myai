class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data -= (
                self.learning_rate *
                parameter.grad
            )

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0


if __name__ == "__main__":
    from tensor import Tensor

    weight = Tensor(1.0)
    target = Tensor(5.0)

    optimizer = SGD(
        [weight],
        learning_rate=0.1
    )

    for step in range(5):
        loss = (weight - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Step {step}: "
            f"weight={weight.data:.4f}, "
            f"loss={loss.data:.4f}"
        )
