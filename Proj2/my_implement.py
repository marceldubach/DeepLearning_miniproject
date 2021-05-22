import torch
import math

torch.set_grad_enabled(False)


class Superclass:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError



class Linear(Superclass):
    def __init__(self, nb_input, nb_output,id):
        std_dev = math.sqrt(2/(nb_input + nb_output))
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.w = torch.empty(nb_output, nb_input).normal_(0, std_dev)  # initialize correct variances
        self.b = torch.empty(nb_output).normal_(0, std_dev)
        self.id = id

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
        s = x @ self.w.t() + self.b # [ N x nb_in] x [ nb_in x nb_out ] + [1 x nb_b]
        return s

    def backward(self, grad_s):
        """
            input:      grad_s (=dl_ds) of size [N x nb_output],
            returns:    grad_w (=dl_dw) [nb_out x nb_in],
                        grad_b (=dl_db) [nb_out],
                        grad_x (=dl_dx) [nb_in]
        """
        if self.grad_w is not None:
            self.grad_w.zero_()
            self.grad_b.zero_()
            self.grad_x.zero_()

        self.grad_w = grad_s.t() @ self.x       # [nb_out x nb_in]
        self.grad_b = grad_s.sum(0)             # [nb_out] sum over samples
        self.grad_x = (grad_s @ self.w).sum(0)  # [N x nb_out] * [nb_out x nb_in], summing over N gives nb_in
        #print("Linear ", self.id, "W:", self.grad_w.std().mean(), "B:", self.grad_b)
        return self.grad_x


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
        #if self.gradient is not None:
            #self.gradient.zero_()
        self.gradient = self.s.sign().add(1).div(2).floor()
        #self.gradient = torch.ones(self.s.size())
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
        #if self.gradient is not None:
            #self.gradient.zero_()
        self.gradient = 1/self.s.cosh().pow(2)
        return self.gradient

class LossMSE:
    def __init__(self):
        self.loss = None
        self.gradient = None

    def forward(self, pred, targ):
        # create one hot matrix
        target = targ.view(targ.size(0),-1)  # add a dimension
        one_hot = torch.empty(target.size(0), pred.size(1), dtype=torch.long).zero_()
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
                dl_dx = module.backward(dl_ds).view(1,-1)

                dl_ds.zero_()
                # print("grad_x shape:", grad_x.size())
            else:
                #print("module ", cnt, " is nonlinear")
                dsigma_ds = module.backward() # [N x nb_out]
                #print("dsigma_ds shape: ", dsigma_ds.size())
                #print("dl_dx shape: ", dl_dx.size())
                dl_ds = dsigma_ds * dl_dx # elementwise [N x nb_out]
                dl_dx.zero_()


    def step(self, eta = 0.1):
        for module in self.modules[::-1]:

            if isinstance(module, Linear):
                # update parameters
                #print("BEFORE UPDATE Linear layer:", module.id, "w", module.w.std().mean().item(), "b", module.b.std().mean().item())
                module.w -= eta * module.grad_w
                module.b -= eta * module.grad_b
                #print("AFTER UPDATE Linear layer:", module.id, "w", module.w.std().mean().item(), "b", module.b.std().mean().item())

def compute_nb_errors(prediction, target):
        nb_errors = 0
        _, predicted_classes = prediction.max(1)
        for k in range(prediction.size(0)):
            if target[k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
        return nb_errors

def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(0, 1) #[0, 1] uniformly distributed
    target = input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target

def generate_hyperplane_set(nb):
    input = torch.empty(nb, 2).uniform_(0, 1) #[0, 1] uniformly distributed
    normal = torch.empty(2).uniform_(0,1) # normal vector
    target = (input @ normal).sign().add(1).div(2).long()
    return input, target

def generate_rectangle_set(nb):
    input = torch.empty(nb, 2).uniform_(-1, 1)  # [0, 1] uniformly distributed
    target = torch.zeros(nb).long()
    x_arr = torch.logical_and(input[:,0]>-0.5, input[:,0]<0.5)
    y_arr = torch.logical_and(input[:, 1] > -0.5, input[:, 1] < 0.5)
    logic = torch.logical_and(x_arr, y_arr)
    target[logic] = 1
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


    train_input, train_target = generate_rectangle_set(nb_samples)
    test_input, test_target = generate_rectangle_set(nb_samples)
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    network = Sequential(    Linear(nb_input,25,1),
                             ReLU(),
                             Linear(25,5,2),
                             ReLU(),
                             Linear(5,nb_output,3)    )


    batch_size = 50
    nb_epochs = 20

    criterion = LossMSE()
    count_step = 0

    for e in range(nb_epochs):
        indexes = torch.randperm(train_input.size(0))
        acc_loss = 0
        errors = 0
        for b in range(0, train_input.size(0), batch_size):
            batch_input = train_input[indexes.narrow(0,b,batch_size)]
            batch_target = train_target[indexes.narrow(0,b,batch_size)]

            # print(x0.narrow(0,b,batch_size))
            batch_output = network.forward(batch_input)

            errors = errors + compute_nb_errors(batch_output, batch_target)
            # print("output", output)
            loss = criterion.forward(batch_output,batch_target)
            dl_dx = criterion.backward()

            network.backward(batch_output, batch_target, dl_dx)
            acc_loss += loss.item()

            network.step()

            count_step += 1

        print("Epoch: ", e, "Loss: ", acc_loss, ", nb errors: ", errors)

    pred = network.forward(test_input.narrow(0,0,20))

