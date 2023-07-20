"""Apply the sbi toolbox
- theta: parameters
- x: output
- number of dimensions in the parameter space: 5
            chl-a
            CDOM
            SPM
            wind speed
            depth
- theta and x:
            torch.Tensor of type float32
            tensor([[p1, p2, p3, p4, p5],
                    p1, p2, p3, p4, p5],
                    ...)"""

# Import libraries

import torch
from sbi.inference import SNPE
from sbi.inference import DirectPosterior

# Number of dimensions in the parameter space
num_dim = 5

# Instantiate the inference object
inference = SNPE()

posterior_estimator = inference.append_simulations(theta, x).train()
posterior = DirectPosterior(posterior_estimator, prior=prior)