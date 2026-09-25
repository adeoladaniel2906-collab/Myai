import json


def save_checkpoint(parameters, filename):
    data = []

    for parameter in parameters:
        data.append(parameter.data)

    with open(filename, "w") as file:
        json.dump(data, file)


def load_checkpoint(parameters, filename):
    with open(filename, "r") as file:
        data = json.load(file)

    if len(data) != len(parameters):
        raise ValueError(
            "Checkpoint parameter count does not match model."
        )

    for parameter, value in zip(parameters, data):
        parameter.data = value
