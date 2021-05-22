import torch
import math

torch.set_grad_enabled(False)


class Superclass:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError



class Linear(Superclass):
    def __init__(self, nb_input, nb_output):
        std_dev = math.sqrt(2/(nb_input + nb_output))
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.w = torch.empty(nb_output, nb_input).normal_(0, std_dev)  # initialize correct variances
        self.b = torch.empty(nb_output).normal_(0, std_dev)

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
        s = x @ self.w.t() + self.b
        return s

    def backward(self, grad_s):
        """
            input:      grad_s (=dl_ds) of size [N x nb_output],
            returns:    grad_w (=dl_dw) [nb_out x nb_in],
                        grad_b (=dl_db) [nb_out],
                        grad_x (=dl_dx) [nb_in]
        """
        self.grad_w = grad_s.t() @ self.x
        self.grad_b = grad_s.sum(0)             # [ nb_out]
        self.grad_x = (grad_s @ self.w).sum(0)  # [N x nb_out] * [nb_out x nb_in], summing over N gives nb_in
        return self.grad_w, self.grad_b, self.grad_x


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
        self.gradient = self.s.sign().add(1).div(2).floor()
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

    def forward(self, pred, targ):
        # create one hot matrix
        target = targ.view(targ.size(0),-1)  # add a dimension
        one_hot = torch.empty(target.size(0), output.size(1), dtype=torch.long).zero_()
        one_hot = one_hot.scatter_(1, target, 1).to(torch.float32) # convert to float

        self.loss = (pred - one_hot).pow(2).mean(dim=0).sum()

        self.gradient = 2 * (pred - one_hot) / pred.size(0)
        return self.loss

    def backward(self):
        return self.gradient


class Sequential(Superclass):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self,x):
        x0 = x
        for module in self.modules:
            x1 = module.forward(x0)
            x0 = x1
        return x1

    def backward(self, output, target, dl_dx, eta=0.1):
        dl_dx = dl_dx.view(output.size(0), -1) # resgaoe to [N x nb_out ]
        dl_ds = dl_dx
        cnt = len(self.modules)
        for module in self.modules[::-1]:  # start from the last one, loop to the first
            cnt -= 1
            if isinstance(module, Linear):
                # print("module ", cnt, " is linear")
                # print("dl_ds shape: ", dl_ds.size())
                module.backward(dl_ds)
                # print("grad_x shape:", grad_x.size())
                dl_dx = module.grad_x.view(1,-1)
                #print("dl_dx shape: ", dl_dx.size())

            else:
                #print("module ", cnt, " is nonlinear")
                dsigma_ds = module.backward() # [N x nb_out]
                #print("dsigma_ds shape: ", dsigma_ds.size())
                #print("dl_dx shape: ", dl_dx.size())
                dl_ds = dsigma_ds * dl_dx # elementwise [N x nb_out]


    def step(self, eta = 0.01):
        # TODO adaptive step size

        for module in self.modules[::-1]:
            if isinstance(module, Linear):
                # update parameters
                module.w -= eta * module.grad_w
                module.b -= eta * module.grad_b
                #print("norm W:", module.grad_w.std().mean())
                #print("norm B:", module.grad_b.std().mean())
                #print(layer)


def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(0, 1) #[0, 1] uniformly distributed
    target = input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target
""" 
Notation:
s1 = w1 * x0 + b1
x1 = sigma(s1)
(in order to have the correct indices ...)
"""
if __name__ == '__main__':
    nb_samples = 1000
    nb_input = 2
    nb_output = 2


    train_input, train_target = generate_disc_set(nb_samples)
    test_input, test_target = generate_disc_set(nb_samples)
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    network = Sequential(    Linear(nb_input,10),
                             ReLU(),
                             Linear(10,5),
                             ReLU(),
                             Linear(5,nb_output)    )
    """
    for module in network.modules:
        print("module:")
        for param in module.param:
            print(param.size())
    """

    batch_size = 50
    nb_epochs = 20

    criterion = LossMSE()

    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0,train_input.size(0),batch_size):
            # print(x0.narrow(0,b,batch_size))
            output = network.forward(train_input.narrow(0,b,batch_size))
            # print("output", output)
            loss = criterion.forward(output, train_target.narrow(0,b,batch_size))
            dl_dx = criterion.backward()

            network.backward(output, train_target.narrow(0,b,batch_size), dl_dx)
            acc_loss += loss.item()
            network.step()

        print("Epoch: ", e, "Loss: ", acc_loss)

    pred = network.forward(test_input.narrow(0,0,20))

