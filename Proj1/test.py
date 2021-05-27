import torch
import torch.nn.functional as F
import torch.nn as nn
from dlc_practical_prologue import generate_pair_sets
""" 
Net_base architecture
global input: N x 2 x 14 x 14 
input to each model: N x 1 x 14 x 14 
    1: convolution layer (k=3)
    N x n1 x 12 x 12
    2: max pooling layer (k=2)
    N x n1 x 6 x 6
    - ReLU
    3: convolution layer 2 (k=3)
    N x n2 x 4 x 4
    4: max pooling layer (k=2)
    N x n2 x 2 x 2
    -ReLU
    -flatten, view as N x (64 x 4 -> 256)
    5: linear layer (256 -> 512)
    N x 512
    - ReLu
        6: concatenate x1 and x2
        N x 1024
        7: linear layer 2 -> (1024 -> 2)
        N x 2
"""
class Net_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1_1 = nn.Linear(256, 512)
        self.fc2_1 = nn.Linear(256, 512)
        self.fc = nn.Linear(2*512, 2)

    def forward(self, x):
        x1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        x1 = F.relu(F.max_pool2d(self.conv1_1(x1), kernel_size=2))
        x1 = F.relu(F.max_pool2d(self.conv1_2(x1), kernel_size=2))
        x2 = F.relu(F.max_pool2d(self.conv2_1(x2), kernel_size=2))
        x2 = F.relu(F.max_pool2d(self.conv2_2(x2), kernel_size=2))
        x1 = x1.view(-1,256)
        x2 = x2.view(-1,256)
        x1 = F.relu(self.fc1_1(x1))
        x2 = F.relu(self.fc2_1(x2))
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)

        return x

""" 
Net_siamese architecture
global input: N x 2 x 14 x 14 
input to each model: N x 1 x 14 x 14 
    1: convolution layer (k=3)
    N x n1 x 12 x 12
    2: max pooling layer (k=2)
    N x n1 x 6 x 6
    - ReLU
    3: convolution layer 2 (k=3)
    N x n2 x 4 x 4
    4: max pooling layer (k=2)
    N x n2 x 2 x 2
    -ReLU
    -flatten, view as N x (64 x 4 -> 256)
    5: linear layer (256 -> 512)
    N x 512
    - ReLu
        6: concatenate x1 and x2
        N x 1024
        7: linear layer 2 -> (1024 -> 2)
        N x 2
"""
class Net_siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(2*512, 2)


    def forward(self,x):
        # split the pair and feed one input at the time to the same architecture
        x1 = x[:,0,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv(x1), kernel_size=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2))
        x1 = x1.view(-1,256)
        x1 = F.relu(self.fc1(x1))

        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        x2 = F.relu(F.max_pool2d(self.conv(x2), kernel_size=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))

        # concatenate outputs and fully connected layer to predict the comparison
        x = torch.cat((x1,x2), dim=1)
        x = self.fc2(x)

        return x

""" 
Net_auxiliary_loss architecture
global input: N x 2 x 14 x 14 
input to each model: N x 1 x 14 x 14 
    1: convolution layer (k=3)
    N x n1 x 12 x 12
    2: max pooling layer (k=2)
    N x n1 x 6 x 6
    - ReLU
    3: convolution layer 2 (k=3)
    N x n2 x 4 x 4
    4: max pooling layer (k=2)
    N x n2 x 2 x 2
    -ReLU
    -flatten, view as N x (64 x 4 -> 256)
    5: linear layer (256 -> 512)
    N x 512
    - ReLu
    6: linear layer 2 -> (512 -> 10)
    N x 10
    7: compute loss with respect to train_classes
        8: concatenate x1 and x2
        N x 1024
        9: linear layer 3 -> (20 -> 2)
        N x 2
        return output and the auxiliary loss 
"""
class Net_aux_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Linear(2*10, 2)

    def forward(self, x, train_classes_batch, criterion):
        x1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        x1 = F.relu(F.max_pool2d(self.conv(x1), kernel_size=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2))
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        x2 = F.relu(F.max_pool2d(self.conv(x2), kernel_size=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)

        # auxiliary loss
        loss = criterion(x1, train_classes_batch[:,0]) + criterion(x2, train_classes_batch[:,1])

        # concatenate outputs and fully connected layer to predict the comparison
        x = torch.cat((x1,x2), dim=1)
        x = self.fc3(x)

        return x, loss

def compute_errors(prediction, target):
    nb_errors = 0
    for i in range(len(prediction)):
        if prediction[i] != target[i]:
            nb_errors += 1
    return nb_errors

def print_train_errors(nb_train_errors, name):
    print('{}: training error {}% for nb_epochs {}'.format(name, 100 * nb_train_errors / (train_target.size(0)), nb_epochs))
    print('Errors at training {}'.format(nb_train_errors))

def print_test_errors(nb_test_errors,name):
    print('{}: testing error {}% for nb_epochs {}'.format(name, 100 * nb_test_errors / (test_target.size(0)), nb_epochs))
    print('Errors at testing {}'.format(nb_test_errors))

def network(model, name):

    # training
    for e in range(nb_epochs):
        acc_loss = 0
        acc_loss_test = 0
        # process input in batches to accelerate process
        for b in range(0, train_input.size(0), batch_size):
            # compute output of the model and auxiliary loss
            if isinstance(model, Net_aux_loss):
                output, aux_loss = model(train_input.narrow(0, b, batch_size), train_classes.narrow(0, b, batch_size),
                                         criterion)
                # sum the loss and the auxiliary loss
                loss = criterion(output, train_target.narrow(0, b, batch_size)) + aux_loss


                output_test, aux_loss_test = model(test_input.narrow(0, b, batch_size), test_classes.narrow(0, b, batch_size),
                                                   criterion)
                # sum the loss and the auxiliary loss
                loss_test = criterion(output_test, test_target.narrow(0, b, batch_size)) + aux_loss_test
            else:
                output = model(train_input.narrow(0, b, batch_size))
                loss = criterion(output, train_target.narrow(0, b, batch_size))

                output_test = model(test_input.narrow(0, b, batch_size))
                loss_test = criterion(output_test, test_target.narrow(0, b, batch_size))

            optimizer.zero_grad()
            # compute gradient with backpropagation
            loss.backward()
            # perform the step to optimize network parameters
            optimizer.step()
            # sum of the loss at each epoch
            acc_loss += loss.item()
            acc_loss_test += loss_test.item()

        print(f"Epoch {e}: Training loss {acc_loss}, Test loss {acc_loss_test}")

    # infer prediction
    _, prediction = torch.max(output, 1)
    # compute number of errors at training
    train_errors = compute_errors(prediction, train_target)
    print_train_errors(train_errors, name)

    # testing
    if isinstance(model, Net_aux_loss):
        output_test, _ = model(test_input, test_classes, criterion)
    else:
        output_test = model(test_input)

    # infer prediction
    _, prediction = torch.max(output_test, 1)
    # compute number of errors at testing
    test_errors = compute_errors(prediction, test_target)
    print_test_errors(test_errors, name)

    return train_errors, test_errors



if __name__ == '__main__':

    rounds = 1
    # prepare data and set optimization parameters
    n_samples = 1000
    batch_size = 100
    nb_epochs = 50
    # load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n_samples)
    # normalize data
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    # define criterion for classification
    criterion = nn.CrossEntropyLoss()

    for model, name in zip([Net_base(), Net_siamese(), Net_aux_loss()], ["simple", "siamese", "auxiliary"]):

        train_errors = []
        test_errors = []
        for i in range(rounds):
            # random seed for each round
            torch.random.seed()

            # reinitialize model to start new round
            model.__init__()
            # choose optimizer ad set learning rate
            eta = 0.05
            optimizer = torch.optim.SGD(model.parameters(), lr=eta)
            # train model
            train_error, test_error = network(model, name)
            train_errors.append(train_error)
            test_errors.append(test_error)
        print("----------------")

