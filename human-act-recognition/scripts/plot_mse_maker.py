#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sys import argv

if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.title("mean squared error vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("mse value")

    with open(argv[1], 'r') as f:
        epochs = []
        mean_squared = []

        for line in f.readlines():
            line = line.strip('\n')
            split = line.split(';')
            epochs.append(float(split[0]))
            mean_squared.append(float(split[1]))

        ax.plot(epochs, mean_squared, label="mse")

    txt='initial mean squared error = ' + str(mean_squared[0]) + '\n  final mean squared error = ' + str(mean_squared[len(mean_squared) -1])
    fig.text(0.25, -.08, txt)
    leg = ax.legend(loc='lower right')
    plt.savefig(argv[2] + '.png', bbox_inches='tight')
    plt.clf()

