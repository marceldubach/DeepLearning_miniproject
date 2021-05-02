import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from dlc_practical_prologue import generate_pair_sets

nb_samples = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(nb_samples)

mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)

all_train_input = train_input.view(-1,14,14)
all_train_labels = train_classes.view(2000)
print(all_train_input.size())
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, 10)


    def forward(self,x):
        x = x.view(-1,1,14,14)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        #x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 2))
        x = x.view(-1,250)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No ReLU here!
        return x


def train_model(model, train_input_2channels, train_labels):
    criterion: CrossEntropyLoss = nn.CrossEntropyLoss()
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)

    train_input = torch.empty(2*train_input_2channels.size(0),14,14)
    train_input[0:train_input_2channels.size(0),:,:] = train_input_2channels[:,0,:,:]
    train_input[1000:2000,:,:] = train_input_2channels[:,1,:,:]

    train_class = torch.empty(2000, dtype=torch.int64)
    train_class[0:1000] = train_labels[:,0]
    train_class[1000:2000] = train_labels[:,1]

    batch_size = 100
    nb_epochs = 200
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0,train_input.size(0),batch_size):
            output = model(train_input.narrow(0,b,batch_size))
            loss = criterion(output, train_class.narrow(0,b,batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()

        print(f"Epoch {e}: Loss {acc_loss}")


def compute_nb_errors(model, test_input, test_target):
    batch_size = 100
    input_0 = test_input[:,0,:,:]
    input_1 = test_input[:, 1, :, :]

    nb_errors = 0

    for i in range(0,test_target.size(0),batch_size):
        output_0 = model(input_0.narrow(0,i,batch_size))
        output_1 = model(input_1.narrow(0,i,batch_size))
        prediction = torch.zeros(batch_size, dtype=torch.int8)
        prediction_0 = output_0.max(1)[1]
        prediction_1 = output_1.max(1)[1]
        for k in range(batch_size):
            if prediction_0[k] > prediction_1[k]:
                prediction[k] = 0
            else:
                prediction[k] = 1
            if prediction[k] != test_target[i+k]:
                nb_errors = nb_errors +1
    return nb_errors

model = Net()
train_model(model, train_input, train_classes)
nb_train_errors = compute_nb_errors(model, train_input, train_target)
nb_test_errors = compute_nb_errors(model, test_input, test_target)

print(f"Training error: {nb_train_errors/1000*100}%")
print(f"Test error: {nb_test_errors/1000*100}%")
