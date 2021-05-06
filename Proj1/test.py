import torch
import torch.nn.functional as F
import torch.nn as nn
from dlc_practical_prologue import generate_pair_sets

import numpy as np #### ONLY FOR STATISTICS, REMOVE FOR HAND IN

""" 
    Net_base architecture
    input: N x 2 x 14 x 14 
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
    6: linear layer 2 -> (512 -> 2)
    N x 2
"""
class Net_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 32, kernel_size=3) #14->12 32x12x12 use Conv3d to swipe across channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) #6->4 64x4x4
        self.fc = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=2)) #32x6x6
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) #64x2x2
        x = x.view(-1,256)
        x = F.relu(self.fc(x))
        x = self.fc2(x)

        return x #return the confidence to be 0 or 1, prediction to infer

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
        6: output1 - output2 
        N x 512
        7: linear layer 2 -> (512 -> 2)
        N x 2
"""
class Net_siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,32,kernel_size=3) # 14->12 32x12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 6->4 64x4x4
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 2)


    def forward(self,x):
        x1 = x[:,0,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv(x1), kernel_size=2)) # 32x6x6
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2)) # 64x2x2
        x1 = x1.view(-1,256)
        x1 = F.relu(self.fc1(x1))

        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        x2 = F.relu(F.max_pool2d(self.conv(x2), kernel_size=2)) # 32x6x6
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2)) # 64x2x2
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))

        x = torch.sub(x1, x2)
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
        5: linear layer (256 -> 200)
        N x 512
        - ReLu
        6: linear layer 2 -> (200 -> 10)
        N x 10
        7: compute loss with respect to train_classes
        8: output1 - output2 
        N x 10
        7: linear layer 3 -> (10 -> 2)
        N x 2
        return x and the auxiliary loss 
"""
class Net_aux_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)


    def forward(self, x, train_classes_batch, criterion):
        x1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        x1 = F.relu(F.max_pool2d(self.conv(x1), kernel_size=2))  # 32x6x6
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2))  # 64x2x2
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        x2 = F.relu(F.max_pool2d(self.conv(x2), kernel_size=2))  # 32x6x6
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2))  # 64x2x2
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)

        loss = criterion(x1, train_classes_batch[:,0]) + criterion(x2, train_classes_batch[:,1])

        x = torch.sub(x1, x2)
        x = self.fc3(x)

        return x, loss

def compute_errors(prediction, target):
    nb_errors = 0
    for i in range(len(prediction)):
        if prediction[i] != target[i]:
            nb_errors += 1
    return nb_errors

def print_train_errors(nb_train_errors):
    print('Training error {}% for nb_epochs {}'.format(100 * nb_train_errors / (train_target.size(0)), nb_epochs))
    print('Errors at training {}'.format(nb_train_errors))

def print_test_errors(nb_test_errors):
    print('Testing error {}% for nb_epochs {}'.format(100 * nb_test_errors / (test_target.size(0)), nb_epochs))
    print('Errors at testing {}'.format(nb_test_errors))

def simple_architecture():

    # set model
    model = Net_base()
    # choose optimizer ad set learning rate
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    # training
    for e in range(nb_epochs):
        acc_loss = 0
        # process input in batches to accelerate process
        for b in range(0, train_input.size(0), batch_size):
            # compute output of the model
            output = model(train_input.narrow(0, b, batch_size))
            # compute loss
            loss = criterion(output, train_target.narrow(0, b, batch_size))
            optimizer.zero_grad()
            # compute gradient with backpropagation
            loss.backward()
            # perform the step to optimize network parameters
            optimizer.step()
            # sum of the loss at each epoch
            acc_loss += loss.item()

        print(f"Epoch {e}: Loss {acc_loss}")
    # infer prediction
    _, prediction = torch.max(output, 1)
    # compute number of errors at training
    train_errors = compute_errors(prediction, train_target)
    print_train_errors(train_errors)

    # testing
    output_test = model(test_input)
    # infer prediction
    _, prediction = torch.max(output_test, 1)
    # compute number of errors at testing
    test_errors = compute_errors(prediction, test_target)
    print_test_errors(test_errors)

    return train_errors, test_errors

def siamese_architecture():

   # set model
    model = Net_siamese()
   # choose optimizer ad set learning rate
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    # training
    for e in range(nb_epochs):
        acc_loss = 0
        # process input in batches to accelerate process
        for b in range(0, train_input.size(0), batch_size):
            # compute output of the model
            output = model(train_input.narrow(0, b, batch_size))
            # compute loss
            loss = criterion(output, train_target.narrow(0, b, batch_size))
            optimizer.zero_grad()
            # compute gradient with backpropagation
            loss.backward()
            # perform the step to optimize network parameters
            optimizer.step()
            # sum of the loss at each epoch
            acc_loss += loss.item()

        print(f"Epoch {e}: Loss {acc_loss}")
    # infer prediction
    _, prediction = torch.max(output, 1)
    # compute number of errors at training
    train_errors = compute_errors(prediction, train_target)
    print_train_errors(train_errors)

    # testing
    output_test = model(test_input)
    # infer prediction
    _, prediction = torch.max(output_test, 1)
    # compute number of errors at testing
    test_errors = compute_errors(prediction, test_target)
    print_test_errors(test_errors)

    return train_errors, test_errors

def auxiliary_loss_architecture():

    # set model
    model = Net_aux_loss()
    # choose optimizer ad set learning rate
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    # training
    for e in range(nb_epochs):
        acc_loss = 0
        # process input in batches to accelerate process
        for b in range(0, train_input.size(0), batch_size):
            # compute output of the model and auxiliary loss
            output, aux_loss = model(train_input.narrow(0, b, batch_size), train_classes.narrow(0, b, batch_size), criterion)
            # sum the loss and the auxiliary loss
            loss = criterion(output, train_target.narrow(0, b, batch_size)) + aux_loss
            optimizer.zero_grad()
            # compute gradient with backpropagation
            loss.backward()
            # perform the step to optimize network parameters
            optimizer.step()
            # sum of the loss at each epoch
            acc_loss += loss.item()

        print(f"Epoch {e}: Loss {acc_loss}")
    # infer prediction
    _, prediction = torch.max(output, 1)
    # compute number of errors at training
    train_errors = compute_errors(prediction, train_target)
    print_train_errors(train_errors)

    # testing
    output_test, _ = model(test_input, test_classes, criterion)
    # infer prediction
    _, prediction = torch.max(output_test, 1)
    # compute number of errors at testing
    test_errors = compute_errors(prediction, test_target)
    print_test_errors(test_errors)

    return train_errors, test_errors



if __name__ == '__main__':
    rounds = 10
    train_errors = []
    test_errors = []
    for i in range(rounds):
        # prepare data and set optimization parameters
        n_samples = 1000
        batch_size = 100
        nb_epochs = 25
        # load data
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n_samples)
        # normalize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)
        # define criterion for classification
        criterion = nn.CrossEntropyLoss()

        torch.random.seed()
        #torch.manual_seed(seed)

        #train_error, test_error = simple_architecture()
        #train_error, test_error = siamese_architecture()
        train_error, test_error = auxiliary_loss_architecture()

        train_errors.append(train_error)
        test_errors.append(test_error)
    print("----------------")
    print("train error % mean {:.02f}%, std {:.02f}%".format(100 * np.mean(train_errors) / (train_target.size(0)), 100 * np.std(train_errors) / (train_target.size(0))))
    print("test error % mean {:.02f}%, std {:.02f}%".format(100 * np.mean(test_errors) / (test_target.size(0)), 100 * np.std(test_errors) / (test_target.size(0)) ))
    print("----------------")

