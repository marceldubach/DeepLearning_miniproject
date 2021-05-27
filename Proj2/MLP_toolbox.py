from torch import empty
import math

class Module:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, nb_input, nb_output):
        std_dev = math.sqrt(2/(nb_input + nb_output))
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.w = empty(nb_output, nb_input).normal_(0, std_dev)  # initialize correct variances
        self.b = empty(nb_output).normal_(0, std_dev)

        self.grad_b = None
        self.grad_w = None
        self.grad_x = None
        self.x = None

    def forward(self, x):
        """
            input:      x of size [N x nb_input]
            returns :   s, where s = w*x + b [N x nb_output]
        """
        self.x = x
        s = x @ self.w.t() + self.b             # [ N x nb_in] x [ nb_in x nb_out ] + [1 x nb_b]
        return s

    def backward(self, grad_s):
        """
            input:      grad_s (=dl_ds) of size [N x nb_output],
            returns:    grad_w (=dl_dw) [nb_out x nb_in],
                        grad_b (=dl_db) [nb_out],
                        grad_x (=dl_dx) [nb_in]
        """
        self.grad_w = grad_s.t() @ self.x       # [nb_out x nb_in]
        self.grad_b = grad_s.sum(0)             # [nb_out] sum over samples
        self.grad_x = (grad_s @ self.w)         # [N x nb_out] * [nb_out x nb_in], summing over N gives nb_in
        return self.grad_x


class ReLU(Module):
    def __init__(self):
        self.param = []
        self.gradient = None    # gradient dl_dxl (derivative w.r.t. the output)
        self.s = None           # s = w*x'+b (input, coming from the linear layer)
        self.x = None           # x = ReLU(s) (activation values)

    def forward(self, s):  # x: [N x nb_input]
        self.s = s
        self.x = s
        self.x[s <= 0] = 0
        return self.x

    def backward(self):
        self.gradient = self.s.sign().add(1).div(2).floor()
        return self.gradient


class Tanh(Module):
    def __init__(self):
        self.param = []
        self.gradient = None
        self.x = None
        self.s = None

    def forward(self,s):
        self.s = s
        self.x = s.tanh()
        return self.x

    def backward(self):
        self.gradient = 1/self.s.cosh().pow(2)
        return self.gradient


class LossMSE(Module):
    def __init__(self):
        self.loss = None
        self.gradient = None

    def forward(self, pred, targ):
        # create one hot matrix
        target = targ.view(targ.size(0),-1)  # add a dimension

        if pred.size(1) > 1:
            # convert to one hot encoding
            one_hot = empty(target.size(0), pred.size(1)).long().zero_()
            one_hot = one_hot.scatter_(1, target, 1).float() # convert to float
            self.loss = (pred - one_hot).pow(2).mean(dim=0).sum()
            self.gradient = 2*(pred- one_hot)/pred.size(0)
        else:
            # if output dimension is only one, no need to convert target to one hot encoding
            self.loss = (pred - target).pow(2).sum()
            # compute the SQUARED error (not normalized by number of samples)
            self.gradient = 2*(pred - target) / pred.size(0)
        return self.loss

    def backward(self):
        return self.gradient


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules
        if not isinstance(self.modules[-1], LossMSE):
            raise NameError('Last module should be instance of LossMSE')
        self.output = None

    def forward(self,x):
        x0 = x
        for module in self.modules[:-1]:
            x1 = module.forward(x0)
            x0 = x1
        self.output = x1
        return self.output

    def loss(self, target):
        loss_module = self.modules[-1]
        loss = loss_module.forward(self.output, target)
        return loss

    def backward(self, eta=0.1):
        loss_module = self.modules[-1]
        dl_dx = loss_module.backward()
        dl_ds = dl_dx

        for module in self.modules[-2::-1]:  # start from the last one, loop to the first
            if isinstance(module, Linear):
                dl_dx = module.backward(dl_ds)
            else:
                dsigma_ds = module.backward() # [N x nb_out]
                dl_ds = dsigma_ds * dl_dx # elementwise [N x nb_out]

    def step(self, eta = 0.1):
        for module in self.modules[-2::-1]:
            if isinstance(module, Linear):  # update the parameter for all linear modules
                module.w -= eta * module.grad_w
                module.b -= eta * module.grad_b