#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sys import argv

if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.title("fitness function vs epoch plot")
    plt.xlabel("epoch")
    plt.ylabel("fitness")

    with open(argv[1], 'r') as f:
        epochs = []
        accuracy = []

        for line in f.readlines():
            line = line.strip('\n')
            split = line.split(';')
            epochs.append(float(split[0]))
            accuracy.append(float(split[2]))

        ax.plot(epochs, accuracy, label="accuracy")

    txt='initial accuracy = ' + str(accuracy[0]) + '\n  final accuracy = ' + str(accuracy[len(accuracy) -1])
    fig.text(0.3, -.08, txt)
    leg = ax.legend(loc='lower right')
    plt.savefig(argv[2] + '.png', bbox_inches='tight')
    plt.clf()

