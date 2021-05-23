from MLP_toolbox import Linear, ReLU, Tanh, LossMSE, Sequential
from torch import empty
from torch import randperm  # used for random permutation for training
import math
import matplotlib.pyplot as plt
import numpy as np


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
    target = input.sub(0.5).pow(2).sum(dim=1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    target = (-target).add(1)

    return input, target


def test_model(network, test_input, test_target, batch_size):
    acc_loss = 0
    nb_errors = 0

    for b in range(0, test_input.size(0), batch_size):

        batch_input = test_input.narrow(0,b,batch_size)
        batch_target= test_target.narrow(0,b,batch_size)

        batch_output = network.forward(batch_input)

        loss = network.loss(batch_target)

        nb_errors = nb_errors + compute_nb_errors(batch_output, batch_target)
        acc_loss = acc_loss + loss.item()

    return acc_loss, nb_errors


def train_model(network, train_input, train_target, nb_epochs, learning_rate, do_test=False, test_input=None, test_target=None):
    batch_size = 50
    error_rec = empty(nb_epochs).zero_()
    loss_rec = empty(nb_epochs).zero_()

    if do_test:
        test_loss_rec = empty(nb_epochs).zero_()
        test_error_rec = empty(nb_epochs).zero_()

    for e in range(nb_epochs):
        indexes = randperm(train_input.size(0))       # random index array without repetition
        acc_loss = 0                                  # accumulated loss for the current epoch
        nb_errors = 0                                 # number of errors for the current epoch

        if do_test:
            test_loss, test_errors = test_model(network, test_input, test_target, batch_size)
            test_loss_rec[e] = test_loss
            test_error_rec[e] = test_errors

        for b in range(0, train_input.size(0), batch_size):
            # retrieve a shuffled batch
            batch_input = train_input[indexes.narrow(0,b,batch_size)]
            batch_target = train_target[indexes.narrow(0,b,batch_size)]

            # model prediction
            batch_output = network.forward(batch_input)

            # accumulate loss and errors
            loss = network.loss(batch_target)
            nb_errors = nb_errors + compute_nb_errors(batch_output, batch_target)
            acc_loss += loss.item()

            # train the model
            network.backward()
            network.step(learning_rate)

        error_rec[e] = nb_errors
        loss_rec[e] = acc_loss

    if do_test:
        return loss_rec, error_rec, test_loss_rec, test_error_rec

    else:
        return loss_rec, error_rec


def plot_results(train_loss_data, train_error_data, test_loss_data=None, test_error_data = None, do_test=False):
    plt.figure()
    plt.title('Training loss', fontsize=18)
    plt.semilogy(np.arange(nb_epochs), train_loss_data[0], '--r', label='LR = 0.01 (train)')
    if do_test:
        plt.semilogy(np.arange(nb_epochs), test_loss_data[0], 'r', label='LR = 0.01 (test)')
    plt.semilogy(np.arange(nb_epochs), train_loss_data[1], '--g', label='LR = 0.05 (train)')
    if do_test:
        plt.semilogy(np.arange(nb_epochs), test_loss_data[1], 'g', label='LR = 0.05 (test)')
    plt.semilogy(np.arange(nb_epochs), train_loss_data[2], '--b', label='LR = 0.1 (train)')
    if do_test:
        plt.semilogy(np.arange(nb_epochs), test_loss_data[2], 'b', label='LR = 0.1 (test)')
    plt.xlabel('Learning epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.ylim((1e1, 1e3))
    plt.legend()
    plt.savefig('loss.png')
    plt.savefig('loss.eps', format='eps')

    plt.figure()
    plt.title('Train and test error rate', fontsize=18)
    plt.plot(np.arange(nb_epochs), train_error_data[0]/nb_samples*100, '--r', label='LR = 0.01 (train)')
    if do_test:
        plt.plot(np.arange(nb_epochs), test_error_data[0]/nb_samples*100, 'r', label='LR = 0.01 (test)')
    plt.plot(np.arange(nb_epochs), train_error_data[1]/nb_samples*100, '--g', label='LR = 0.05 (train)')
    if do_test:
        plt.plot(np.arange(nb_epochs), test_error_data[1]/nb_samples*100, 'g', label='LR = 0.05 (test)')
    plt.plot(np.arange(nb_epochs), train_error_data[2]/nb_samples*100, '--b', label='LR = 0.1 (train)')
    if do_test:
        plt.plot(np.arange(nb_epochs), test_error_data[2]/nb_samples*100, 'b', label='LR = 0.1 (test)')
    plt.xlabel('Learning epoch', fontsize=16)
    plt.ylabel('Error rate [%]', fontsize=16)
    plt.legend()
    plt.savefig('error_rate.png')
    plt.savefig('error_rate.eps', format='eps')

    print("Saved results as 'loss.png' and error_rate.png'")

def visualize_prediction(network, test_input, test_iput_original):

    batch_size = 50
    plt.figure()
    nb_errors = 0
    for b in range(0,test_input.size(0), batch_size):
        batch_input = test_input.narrow(0,b,batch_size)
        batch_input_original = test_iput_original.narrow(0,b,batch_size)
        batch_output = network.forward(batch_input)


        batch_target = test_target.narrow(0,b,batch_size)
        nb_errors = nb_errors + compute_nb_errors(batch_output, batch_target)

        batch_output = batch_output.view(batch_output.size(0))
        pred1 = batch_input_original[batch_output >= 0.5]
        pred0 = batch_input_original[batch_output < 0.5]

        plt.plot(pred1[:,0].data.numpy(), pred1[:,1].data.numpy(), 'ro')
        plt.plot(pred0[:,0].data.numpy(), pred0[:,1].data.numpy(), 'bo')

    plt.title('MLP predictions', fontsize=18)
    plt.legend(['class 1', 'class 0'])
    plt.xlabel(r'$x_1$', fontsize=16)
    plt.ylabel(r'$x_2$', fontsize=16)
    plt.axis('equal')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig('MLP_prediction.png')
    plt.savefig('MLP_prediction.eps', format='eps')

    print("Error rate: {:.2f}%".format(nb_errors/nb_samples*100))
    print("Saved visualization of model prediction to 'MLP_prediction.png'")


if __name__ == '__main__':
    nb_samples = 1000
    nb_input = 2
    nb_output = 1

    nb_epochs = 25
    learning_rates = [0.01, 0.05, 0.1]
    do_test = True

    # generate and normalize the data
    train_input_original, train_target = generate_disc_set(nb_samples)
    test_input_original, test_target = generate_disc_set(nb_samples)
    mu, std = train_input_original.mean(dim=0), train_input_original.std()
    train_input = train_input_original.sub(mu).div(std)
    test_input = test_input_original.sub(mu).div(std)

    train_loss_data = empty(len(learning_rates), nb_epochs).zero_()
    train_error_data = empty(len(learning_rates), nb_epochs).zero_()
    if do_test:
        test_loss_data = empty(len(learning_rates), nb_epochs).zero_()
        test_error_data = empty(len(learning_rates), nb_epochs).zero_()


    for i in range(len(learning_rates)):
        # create an MLP
        network = Sequential(Linear(nb_input, 25),
                             ReLU(),
                             Linear(25, 25),
                             ReLU(),
                             Linear(25, 25),
                             ReLU(),
                             Linear(25, nb_output),
                             Tanh(),
                             LossMSE())

        lr = learning_rates[i]
        if do_test:
            train_loss_rec, train_err_rec, test_loss_rec, test_err_rec = train_model(network, train_input, train_target, nb_epochs, lr, do_test, test_input, test_target)


            test_loss_data[i] = test_loss_rec
            test_error_data[i] = test_err_rec
        else:
            train_loss_rec, train_err_rec = train_model(network, train_input, train_target, nb_epochs, lr, do_test, test_input, test_target)

        train_loss_data[i] = train_loss_rec
        train_error_data[i] = train_err_rec

    if do_test:
        plot_results(train_loss_data, train_error_data, test_loss_data, test_error_data, do_test)

    # Train a final model over 25 epochs
    network = Sequential(Linear(nb_input, 25),
                         ReLU(),
                         Linear(25, 25),
                         ReLU(),
                         Linear(25, 25),
                         ReLU(),
                         Linear(25, nb_output),
                         Tanh(),
                         LossMSE())

    train_loss_rec, train_err_rec = train_model(network, train_input, train_target, nb_epochs, lr)
    visualize_prediction(network, test_input, test_input_original)
