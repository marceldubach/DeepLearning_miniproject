# DL_miniproject

Repository for Deep Learning course project

Part 1: MNIST classification & comparison
Proj1/test.py implements feed-forward neural networks that compare two MNIST images against each other and output a binary value depending on the value which is bigger.

The network architectures compared are:
- separately learned classifiers (conv_net), followed by one linear layer for the comparison (Net_base)
- siamese layers for classifiers, followed by one linear layer for the comparison (Net_siamese)
- siamese layers followed by a linear layer, trained with auxiliary loss (Net_aux_loss)

Part 2: toolbox for FFNN-networks
The toolbox implements a set of modules (torch.nn.Module), notably
- linear layer
- relu layer
- tanh layer
- MSELoss
- Sequential network

The module used to learn a classifier for a toy dataset:
Inputs lie in the intercal [0, 1] x [0, 1]
Targets are 1 if the point is contained in a circle centered at [0.5,0.5] and 0 else 

