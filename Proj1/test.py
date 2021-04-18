import torch
import torch.nn.functional as F
import torch.nn as nn
from dlc_practical_prologue import generate_pair_sets

""" 
    input: N x 14 x 14
    1 convolution layer (k=5)
    N x n1 x 10 x 10
    2 pooling layer (2)
    N x n1 x 5 x 5
    view as N x (n1  x 25)
    - ReLu
    3 linear layer -> auxiliary output (250 -> 100)
    N x 100
    - ReLu
    5 linear layer -> (100 -> 10)
    N x 10
"""


n_samples = 1000
batch_size = 100
nb_epochs =  1
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n_samples)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,10,kernel_size=5)
        self.fc1 = nn.Linear(250,100)
        self.fc2 = nn.Linear(100,10)


    def forward(self,x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x0 = F.relu(F.max_pool2d(self.conv(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1,250)
        x0 = F.relu(self.fc1(x0))
        x0 = self.fc2(x0)

        x1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1,250)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        x0_pred = x0.max(1)[1]
        x1_pred = x1.max(1)[1]

        x  = F.tanh(x0_pred - x1_pred)
        return x

criterion = nn.CrossEntropyLoss()

model = Net()
eta = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = eta)
#for e in range(nb_epochs)
for e in range(nb_epochs):
    for b in range(0,train_input.size(0),batch_size):
        output = model(train_input.narrow(0,b,batch_size))
        print(output.size())
        print(output)
        loss = criterion(output, train_target.narrow(0,b,batch_size))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_output = model(train_input)
print(train_target[0:30])
