import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    loss_t_base = pd.read_csv('simpletraining_loss.csv')
    loss_t_siam = pd.read_csv('siamesetraining_loss.csv')
    loss_t_aux = pd.read_csv('auxiliarytraining_loss.csv')
    loss_ts_base = pd.read_csv('simpletesting_loss.csv')
    loss_ts_siam = pd.read_csv('siamesetesting_loss.csv')
    loss_ts_aux = pd.read_csv('auxiliarytesting_loss.csv')
   
    # plot loss evolution at training and testing
    plt.figure(figsize=(8,5))
    x = np.linspace(1,25,25)
    plt.plot(x, loss_t_base['loss_train'], '--')
    plt.plot(x, loss_t_siam['loss_train'], '--')
    plt.plot(x, loss_t_aux['loss_train'], '--')
    plt.plot(x, loss_ts_base['loss_test'], 'b' )
    plt.plot(x, loss_ts_siam['loss_test'], 'orange')
    plt.plot(x, loss_ts_aux['loss_test'], 'g')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["TR base architecture", "TR siamese like", "TR auxiliary loss", "TS base architecture", "TS siamese like", "TS auxiliary loss"])
    plt.title("Loss evolution through epochs")
    
    # plot performances with 30 rounds 
    stat_base = pd.read_csv('simple_statistics.csv') 
    stat_siam = pd.read_csv('siamese_statistics.csv')
    stat_aux = pd.read_csv('auxiliary_statistics.csv')
    plt.figure(figsize=(8,5))
    x = ["base train", "base test", "siamese train", "siamese test", "auxiliary train", "auxiliary test"]
    y = [stat_base['train_mean'][0], stat_base['test_mean'][0], stat_siam['train_mean'][0], stat_siam['test_mean'][0], stat_aux['train_mean'][0], stat_aux['test_mean'][0] ]
    e = [stat_base['train_std'][0], stat_base['test_std'][0], stat_siam['train_std'][0], stat_siam['test_std'][0], stat_aux['train_std'][0], stat_aux['test_std'][0] ]
    for i in range(3):
        plt.errorbar(x[2*i:2*(i+1)], y[2*i:2*(i+1)], e[2*i:2*(i+1)], fmt='-o', capsize=5, mfc='red' )
    plt.xlabel("architectures")
    plt.ylabel("error %")
    plt.legend(["Base architecture", "Weight sharing", "Auxiliary loss"])
    plt.title("Performance estimates through 30 rounds")
