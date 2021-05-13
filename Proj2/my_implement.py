import torch
import math

torch.set_grad_enabled(False)


class Superclass:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Superclass):
    def __init__(self, nb_input, nb_output):
        std_dev = math.sqrt(2/(nb_input + nb_output))
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.w = torch.empty(nb_output, nb_input).normal_(0, std_dev)  # initialize correct variances
        self.b = torch.empty(nb_output).normal_(0, std_dev)
        self.parameters = [self.w, self.b]

        self.grad_b = None
        self.grad_w = None
        self.grad_x = None
        self.x = None
        self.s = None

    def forward(self, x):
        """
            input:      x of size [N x nb_input]
            returns :   s, where s = w*x + b [N x nb_output]
        """
        self.x = x
        self.s = x @ self.w.t() + self.b
        return self.s

    def backward(self, grad_s):
        """
            input:      grad_s (=dl_ds) of size [N x nb_output],
            returns:    grad_w (=dl_dw) [nb_out x nb_in],
                        grad_b (=dl_db) [nb_out],
                        grad_x (=dl_dx) [nb_in]
        """
        self.grad_w = grad_s.t() @ self.x
        self.grad_b = grad_s.sum(0)             # sum over all samples
        self.grad_x = (grad_s @ self.w).sum(0)  # [N x nb_out] * [nb_out x nb_in], summing over N gives nb_in
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
        self.gradient = None    # gradient dl_dxl (derivative w.r.t. the output)
        self.s = None           # s = w*x'+b (input, coming from the linear layer)
        self.x = None           # x = ReLU(s) (activation values)

    def forward(self, s):  # x: [N x nb_input]
        self.s = s
        self.x = self.activation_function(s)
        return self.x

    def backward(self, y):
        self.gradient = self.s.sign().add(1).div(2)
        return self.gradient

    def activation_function(self, s):
        x = s
        x[s <= 0] = 0
        return x


"""
class Tanh(Superclass):
    def __init__(self):
        self.param = None

    def forward(self):

    def backward(self):

    def activation(self):
"""

# class MSELoss(Superclass):


class Sequential(Superclass):
    def __init__(self, modules):
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, loss):
        for module in self.modules[::-1]:  # start from the last one, loop to the first
            grad = module.backward(loss)  # a revoir...


""" PSEUDOCODE
network = Sequential( (Linear(n_input, n_output), ReLU(),..) )

output = netword.forward()
loss = MSELoss.computeloss(output, ...)

"""

""" 
Notation:
s1 = w1 * x0 + b1
x1 = sigma(s1)
(in order to have the correct indices ...)
"""
if __name__ == '__main__':
    nb_input = 10
    nb_output = 2
    linear = Linear(nb_input, nb_output)
    relu = ReLU()
    x0 = torch.empty(100, nb_input).uniform_()
    s1 = linear.forward(x0)
    x1 = relu.forward(s1)

    dl_dx1 = torch.empty(nb_output).uniform_()
    dl_ds1 = relu.backward(dl_dx1)
    dl_dw, dl_db, dl_dx0 = linear.backward(dl_ds1)
    # TODO implement linear.backward -> dl_dx
    print('x0:', x0.size())
    print('s1:', s1.size())
    print('x1:', x1.size())

    print('dl_dx0:', dl_dx0.size())
    print('dl_ds0:', dl_ds1.size())
    print('dl_dx1:', dl_dx1.size())

    print('dl_dw:', dl_dw.size())
    print('dl_db0', dl_db.size())
