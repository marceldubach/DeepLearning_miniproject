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
        self.param = [self.w, self.b]

        self.grad_b = None
        self.grad_w = None
        self.grad_x = None
        self.x = None
        #self.s = None

    def forward(self, x):
        """
            input:      x of size [N x nb_input]
            returns :   s, where s = w*x + b [N x nb_output]
        """
        self.x = torch.empty(x.size()).zero_()
        s = x @ self.w.t() + self.b
        print("self.x stored")
        return s

    def backward(self, grad_s):
        """
            input:      grad_s (=dl_ds) of size [N x nb_output],
            returns:    grad_w (=dl_dw) [nb_out x nb_in],
                        grad_b (=dl_db) [nb_out],
                        grad_x (=dl_dx) [nb_in]
        """
        print("size self.x", self.x.size())
        print("size grad_s", grad_s.size())
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
        self.x = s
        self.x[s <= 0] = 0
        return self.x

    def backward(self):
        self.gradient = self.s.sign().add(1).div(2)
        return self.gradient


class Tanh(Superclass):
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


class LossMSE:
    def __init__(self):
        self.loss = None
        self.gradient = None

    def compute_loss(self, output, target):
        # create one hot matrix
        target = target.view(target.size(0),-1)  # add a dimension
        one_hot = torch.empty(target.size(0), output.size(1), dtype=torch.long).zero_()
        one_hot = one_hot.scatter_(1, target, 1).to(torch.float32) # convert to float
        #  print("one hot", one_hot)
        #  print("target", target)
        sample_loss = (one_hot - output).norm(dim=1)
        self.loss = sample_loss.sum()
        #  print("loss ", self.loss)
        sample_loss = sample_loss.view(output.size(0), 1)
        self.gradient = (output - one_hot).div(sample_loss)
        return self.loss

    def get_gradient(self):
        return self.gradient


class Sequential(Superclass):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self,x):
        x0 = x
        for module in self.modules:
            if isinstance(module, Linear):
                # initialize the output tensor
                x1 = torch.empty(x0.size(0), module.nb_output).zero_()
            else:
                x1 = torch.empty(x0.size())

                #print("bmin: ", b, "bmax: ", b + batch_size)
                #print("input shape" ,x0.narrow(0,b,batch_size).size())
                #print("output shape", x1[b:b+batch_size,:].size())
            x1 = module.forward(x0)

            x0 = x1
        return x1

    def backward(self, output, target, dl_dx):
        dl_dx = dl_dx.view(output.size(0), -1) # resgaoe to [N x nb_out ]
        dl_ds = dl_dx
        cnt = len(self.modules)
        for module in self.modules[::-1]:  # start from the last one, loop to the first
            cnt -= 1
            if isinstance(module, Linear):
                print("module ", cnt, " is linear")

                print("dl_ds shape: ", dl_ds.size())
                grad_w, grad_b, grad_x = module.backward(dl_ds)
                print("grad_x shape:", grad_x.size())
                dl_dx = grad_x.view(1,-1)
                print("dl_dx shape: ", dl_dx.size())
                # update parameters (TODO: choose step size)
                module.param[0] -= grad_w
                module.param[1] -= grad_b

            else:
                print("module ", cnt, " is nonlinear")
                dsigma_ds = module.backward() # [N x nb_out]
                print("dsigma_ds shape: ", dsigma_ds.size())
                print("dl_dx shape: ", dl_dx.size())
                dl_ds = dsigma_ds * dl_dx # elementwise [N x nb_out]

""" 
Notation:
s1 = w1 * x0 + b1
x1 = sigma(s1)
(in order to have the correct indices ...)
"""
if __name__ == '__main__':
    nb_samples = 100
    nb_input = 10
    nb_output = 2
    x0 = torch.empty(nb_samples, nb_input).uniform_()
    target = torch.randint(0, 2, (nb_samples,))


    network = Sequential(    Linear(nb_input,8),
                             ReLU(),
                             Linear(8,5),
                             ReLU(),
                             Linear(5,nb_output)    )
    """
    for module in network.modules:
        print("module:")
        for param in module.param:
            print(param.size())
    """

    batch_size = 50
    acc_loss = 0
    criterion = LossMSE()
    for b in range(0,x0.size(0),batch_size):
        output = network.forward(x0.narrow(0,b,batch_size))
        loss = criterion.compute_loss(output, target.narrow(0,b,batch_size))
        dl_dx = criterion.get_gradient()
        network.backward(output, target.narrow(0,b,batch_size), dl_dx)
        acc_loss += loss.item()

    print("Loss: ", acc_loss)

