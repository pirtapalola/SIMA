import torch
import numpy as np
from models.modelSBI import AmortizedPosterior


def test_forward():
    # Set up the neural network instance for testing
    input_dim = 5
    output_dim = 150
    hidden_dim = 256
    amortized_net = AmortizedPosterior(input_dim, output_dim, hidden_dim)

    # Test the forward pass of the network
    input_data = torch.tensor(np.random.rand(10, input_dim), dtype=torch.float32)
    output = amortized_net(input_data)
    assert output.shape == (10, output_dim)  # Check if output shape is as expected
