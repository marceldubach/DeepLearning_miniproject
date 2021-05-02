import torch
import torch.nn.functional as F
import torch.nn as nn
from dlc_practical_prologue import generate_pair_sets

""" 
    global input: N x 2 x 14 x 14
    input to the model: N x 1 x 14 x 14
    1 convolution layer (k=3)
    N x n1 x 12 x 12
    2 pooling layer (2)
    N x n1 x 6 x 6
    - ReLU
    3 convolution layer (k=3)
    N x n2 x 4 x 4
    4 pooling layer (2)
    N x n2 x 2 x 2
    -ReLU
    view as N x (n2 x 4 -> 256)
    5 linear layer -> auxiliary output (256 -> 200)
    N x 200
    - ReLu
    6 linear layer -> (200 -> 10)
    N x 10
"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,32,kernel_size=3) # 14->12 32x12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 6->4 64x4x4
        self.fc1 = nn.Linear(256,200)
        self.fc2 = nn.Linear(200,10)


    def forward(self,x):
        x1 = x[:,0,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv(x1), kernel_size=2)) # 32x6x6
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2)) # 64x2x2
        x1 = x1.view(-1,256)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        x2 = F.relu(F.max_pool2d(self.conv(x2), kernel_size=2)) # 32x6x6
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2)) # 64x2x2
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)

        return x1, x2


def compute_nb_errors(model, data_input, data_target):
    batch_size = 100
    nb_errors = 0
    for b in range(0, data_input.size(0), batch_size):
        output1, output2 = model(data_input.narrow(0, b, batch_size))
        _, predicted_classes_1 = torch.max(output1, 1)
        _, predicted_classes_2 = torch.max(output2, 1)
        for i in range(batch_size):
            #print("pred1 {} vs train_class {}".format(predicted_classes_1[i], data_target[(b + i, 0)]))
            #print("pred2 {} vs train_class {}".format(predicted_classes_2[i], data_target[(b + i, 1)]))
            if data_target[(b + i, 0)] != predicted_classes_1[i]:
                nb_errors = nb_errors + 1
            if data_target[(b + i, 1)] != predicted_classes_2[i]:
                nb_errors = nb_errors + 1
    return nb_errors

def compute_comparison_nb_errors(model, data_input, data_target):
    batch_size = 100
    nb_errors = 0
    for b in range(0, data_input.size(0), batch_size):
        output1, output2 = model(data_input.narrow(0, b, batch_size))
        _, predicted_classes_1 = torch.max(output1, 1)
        _, predicted_classes_2 = torch.max(output2, 1)
        # operation to compare if second digit is greater than first digit
        comp = predicted_classes_2 - predicted_classes_1
        mask = comp >= 0
        # if first digit is lesser or equal to the second one, the target returns 1
        comp[mask] = 1
        # otherwise it returns 0, ReLU brings all negative numbers to 0
        comp = F.relu(comp)
        for i in range(batch_size):
            #print("comp {} vs train_target {}".format(comp[i], data_target[b + i]))
            if data_target[b + i] != comp[i]:
                nb_errors = nb_errors + 1
    return nb_errors

if __name__ == '__main__':
    n_samples = 1000
    batch_size = 100
    nb_epochs = 25
    # load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n_samples)
    # normalize data
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    criterion = nn.CrossEntropyLoss()
    model = Net()
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)

    # training
    for e in range(nb_epochs):
        for b in range(0,train_input.size(0),batch_size):
            output1, output2 = model(train_input.narrow(0,b,batch_size))
            # sum of each loss
            loss = criterion(output1, train_classes[:,0].narrow(0,b,batch_size)) + criterion(output2, train_classes[:,1].narrow(0,b,batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # computing performance
    nb_errors = compute_nb_errors(model, train_input, train_classes)
    print('Training error {}% for nb_epochs {}'.format(100*nb_errors/(2*train_input.size(0)), nb_epochs))
    print('digits misclassified at training {}'.format(nb_errors))
    nb_errors_comp = compute_comparison_nb_errors(model, train_input, train_target)
    print('Training comparison error {}% for nb_epochs {}'.format(100 * nb_errors_comp / (train_input.size(0)), nb_epochs))
    print('comparison errors at training  {}'.format(nb_errors_comp))
    test_errors = compute_nb_errors(model, test_input, test_classes)
    print('Testing error {}% for nb_epochs {}'.format(100 * test_errors / (2 * test_input.size(0)), nb_epochs))
    print('digits misclassified at testing {}'.format(test_errors))
    test_errors_comp = compute_comparison_nb_errors(model, test_input, test_target)
    print('Testing comparison error {}% for nb_epochs {}'.format(100 * test_errors_comp / (test_input.size(0)),nb_epochs))
    print('comparison errors at testing {}'.format(test_errors_comp))

    # print to check the targeted class for a the desired digits comparison
    #print("test_classes {}".format(test_classes[:10]))
    #print("test_targets {}".format(test_target[:10]))

