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
        self.parameters = [self.weight, self.bias]

        self.grad_b = None
        self.grad_w = None
        self.grad_x = None
        self.output = None

    def forward(self,x):
        """ input [N x nb_input], self.output: [N x nb_output] """
        self.input = x
        self.output = x @ self.weight.t() + self.bias
        return self.output

    def backward(self, grad_output):
        """ grad_output [N x nb_output], self.input [N x nb_input], returns [ nb_output, nb_input ] """
        self.grad_w = grad_output.t() @ self.input
        self.grad_b = grad_output.sum(0)

        """ weight [ n_out x n_in ]"""
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
        self.gradient = None
        self.activation = None

    def forward(self,x): # x: [N x nb_input]
        self.activation = self.activation_function(x)
        return self.activation

    def backward(self,y):
        self.gradient = self.activation.sign().add(1).div(2)
        return self.gradient

    def activation_function(self,x):
        self.activation = x
        self.activation[ x <= 0 ] = 0
        return self.activation

"""
class Tanh(Superclass):
    def __init__(self):
        self.param = None

    def forward(self):

    def backward(self):

    def activation(self):
"""

#class MSELoss(Superclass):

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
    dl_ds0 = relu.backward(dl_dx1)
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



