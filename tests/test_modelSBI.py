import torch
import torch.nn as nn
import pytest
from models.neural_network import SimpleNeuralNetwork  # Import your neural network module

@pytest.fixture
def neural_net():
    input_dim = 150
    hidden_dim = 256
    output_dim = 5
    return SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)


def test_forward_pass(neural_net):
    batch_size = 32
    input_data = torch.randn(batch_size, neural_net.input_dim)
    output = neural_net(input_data)
    expected_output_shape = (batch_size, neural_net.output_dim)

    assert output.shape == expected_output_shape


def test_output_range(neural_net):
    batch_size = 32
    input_data = torch.randn(batch_size, neural_net.input_dim)
    output = neural_net(input_data)

    # Check if output values are within a reasonable range (e.g., -10 to 10)
    assert torch.all(output >= -10)
    assert torch.all(output <= 10)


def test_gradients(neural_net):
    batch_size = 32
    input_data = torch.randn(batch_size, neural_net.input_dim)
    target_data = torch.randn(batch_size, neural_net.output_dim)

    optimizer = torch.optim.SGD(neural_net.parameters(), lr=0.1)

    # Compute loss and backpropagate
    optimizer.zero_grad()
    output = neural_net(input_data)
    loss = nn.MSELoss()(output, target_data)
    loss.backward()

    # Check if gradients are non-zero after backward pass
    for param in neural_net.parameters():
        assert param.grad is not None
        assert torch.sum(param.grad) != 0
