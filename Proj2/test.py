from MLP_toolbox import Linear, ReLU, Tanh, LossMSE, Sequential
from torch import empty
from torch import randperm  # used for random permutation for training
import math


def compute_nb_errors(prediction, target):
    """
    compute
    :param prediction [N x nb_output]:  tensor containing the model output
    :param target [N], tensor containing the desired target
    :return: number of errors.

    - If the network output is multidimensional, the predicted value corresponds to the unit with maximal activation
    - If the network is one dimensional, the absolute value of abs(prediction-target) should by < 0.5
    """
    nb_errors = 0
    if prediction.size(1)>1:
        # MLP has several units in output layer (use one hot encoding)
        _, predicted_classes = prediction.max(1)
        for k in range(prediction.size(0)):
            if target[k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    else:
        # prediction has only 1 output (no one hot encoding)
        for k in range(prediction.size(0)):
            if (prediction[k] - target[k]).abs() >= 0.5:
                nb_errors = nb_errors + 1
    return nb_errors


def generate_disc_set(nb_samples):
    """
    :param nb_samples: number of data samples to create
    generate a dataset that contains uniformly distributed samples in [0,1] x [0,1]
    target value is 1 if the distance from the point [0.5], [0.5] is < 1/(2*math.pi)
    NOTE: this dataset is balanced when samples are uniformly distributed
    """
    input = empty((nb_samples, 2)).uniform_(0,1)
    target = input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target


if __name__ == '__main__':

    # given in the problem statement
    nb_samples = 1000
    nb_input = 2
    nb_output = 1

    # training parameters
    batch_size = 50
    nb_epochs = 50
    learning_rate = 0.05

    # generate and normalize the data
    train_input, train_target = generate_disc_set(nb_samples)
    test_input, test_target = generate_disc_set(nb_samples)
    mu, std = train_input.mean(dim=0), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    # create an MLP
    network = Sequential(    Linear(nb_input,25),
                             ReLU(),
                             Linear(25,25),
                             ReLU(),
                             Linear(25,25),
                             ReLU(),
                             Linear(25,nb_output),
                             Tanh(),
                             LossMSE())

    for e in range(nb_epochs):

        indexes = randperm(train_input.size(0))       # random index array without repetition
        acc_loss = 0                                        # accumulated loss for the current epoch
        errors = 0                                          # number of errors for the current epoch

        for b in range(0, train_input.size(0), batch_size):
            # retrieve a shuffled batch
            batch_input = train_input[indexes.narrow(0,b,batch_size)]
            batch_target = train_target[indexes.narrow(0,b,batch_size)]

            # model prediction
            batch_output = network.forward(batch_input)

            # accumulate loss and errors
            loss = network.loss(batch_target)
            errors = errors + compute_nb_errors(batch_output, batch_target)
            acc_loss += loss.item()

            # train the model
            network.backward()
            network.step(learning_rate)

        print("Training epoch {:d}: \t loss value: {:.2f}, \t error rate: {:.2f}%".format(e, acc_loss, errors/nb_samples*100))


