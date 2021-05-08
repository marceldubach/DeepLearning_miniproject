from torch import empty
import math

def generate_disc_set(nb):
    input = empty(nb, 2).uniform_(0, 1) #[0, 1] uniformly distributed
    target = input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target


if __name__ == '__main__':
    nb_samples = 1000
    train_input, train_target = generate_disc_set(nb_samples)
    test_input, test_target = generate_disc_set(nb_samples)
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)