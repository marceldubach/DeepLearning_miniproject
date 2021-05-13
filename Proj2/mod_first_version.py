import torch
import math

class Superclass():
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def param(self):
        return []

class Linear(Superclass):
    def __init__(self, nb_input, nb_output):
        std_dev = math.sqrt(2/(nb_input + nb_output))
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.weight = torch.empty(nb_output, nb_input).normal_(0, std_dev) # initialize correct variances
        self.bias = torch.empty(nb_output).normal_(0, std_dev)

        self.grad_b = None
        self.grad_w = None
        self.grad_x = None
        self.output = None
        self.parameters = [[self.weight, self.grad_w],  [self.bias, self.grad_b]]

    def forward(self,x):
        """ input [N x nb_input], self.output: [N x nb_output] """
        self.input = x
        self.output = x @ self.weight.t() + self.bias
        return self.output

    def backward(self, grad_output):
        """ grad_output [N x nb_output], self.input [N x nb_input], grad_w [ nb_output, nb_input ], grad_b [ nb_output ] """
        self.grad_w = grad_output.t() @ self.input
        self.grad_b = grad_output.sum(0)

        """ weight [ nb_out x nb_in ], grad_x [ nb_in ] """
        self.grad_x = (grad_output @ self.weight).sum(0)

        return self.grad_w, self.grad_b, self.grad_x

    def param(self):
        return self.parameters

"""
Sequential ( Linear(...) , ReLU(), .. Linear()
self.modules [ ... ]
"""
class ReLU(Superclass):
    def __init__(self):
        self.param = []
        self.grad_act = None
        self.activation = None

    def forward(self, x): # x: [N x nb_input]
        self.activation = self.activation_function(x)
        return self.activation

    def backward(self):
        #self.gradient = self.activation.sign().add(1).div(2)
        self.grad_act = self.activation.sign().add(1).div(2).floor() # added floor to have 0 for 0 entries instead of 0.5
        return self.grad_act

    def activation_function(self,x):
        self.activation = x
        self.activation[ x <= 0 ] = 0
        return self.activation


class Tanh(Superclass):
    def __init__(self):
        self.param = []
        self.grad_act = None
        self.activation = None

    def forward(self, x):
        self.activation = self.activation_function(x)
        return self.output

    def backward(self):
        self.grad_act = 1 - self.activation.pow(2)
        return grad_act

    def activation_function(self, x):
        self.activation = x
        self.activation = ((2*x).exp() - 1)/((2*x).exp() + 1)
        return self.activation

"""
class MSELoss(Superclass):
    def __init__(self):
        self.grad_act = None
        self.grad_x = None
        self.grad_w = None
        self.grad_b = None
        self.param = []

    def forward(self, pred, targ):
        self.loss = (pred - targ).pow(2).mean()
        return self.loss

    def backward(self, pred, targ, grad_act):
        self.grad_x = 2*(pred - targ)/pred.size(0)
        self.grad_act = grad_act * self.grad_x
        return grad_x, grad_act
"""

class Sequential(Superclass):
    def __init__(self, modules):
        self.modules = modules

    def forward(self):
        for module in self.Modules:
            x = module.forward(x)
        return x

    def backward(self, loss):
        for module in self.Modules[::-1]: # start from the last one, loop to the first
            grad = module.backward(loss) # a revoir...

""" PSEUDOCODE
network = Sequential( (Linear(n_input, n_output), ReLU(),..) )

output = netword.forward()
loss = MSELoss.computeloss(output, ...)

"""

if __name__=='__main__':
    nb_input = 10
    nb_output = 3
    linear = Linear(nb_input, nb_output)
    relu = ReLU()
    x0 = torch.empty(100,10).uniform_(0,1)
    s0 = linear.forward(x0)
    x1 = relu.forward(s0)

    dl_dx1 = torch.empty(nb_output).uniform_(0,1)
    dl_ds0 = relu.backward()
    dl_dw, dl_db, dl_dx0 = linear.backward(dl_ds0)
    # TODO implement linear.backward -> dl_dx
    print('x0:', x0.size())
    print('s1:', s0.size())
    print('x1:', x1.size())

    print('dl_dx0:', dl_dx0.size())
    print('dl_ds0:', dl_ds0.size())
    print('dl_dx1:', dl_dx1.size())

    print('dl_dw:', dl_dw.size())
    print('dl_db0', dl_db.size())



