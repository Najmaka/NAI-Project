#!/usr/bin/python3

from sys import argv
import re


def read_file(path):
    content = []
    with open(path, 'r') as f:
        lines = list(f)
        for line in lines:
            content.append(line.strip('\n'))
    return content


if __name__ == '__main__':
    if len(argv) != 5:
        print("provide path to features, activity_labels, train_features and train_labels")
        exit(1)

    features = read_file(argv[1])
    instances = read_file(argv[3])
    labels = read_file(argv[4])
    activity_labels = [x.split(' ') for x in read_file(argv[2])]
    labels_dict = dict()
    for al in activity_labels:
        labels_dict[al[0]] = al[1]

    if len(instances) != len(labels):
        raise Exception('Feature and Labels length mismatch')

    arff = ['@RELATION HumanAct\n']
    for line in features:
        feature = line.split(' ')
        attr_val = '{}_{}'.format(feature[0], re.sub('[^0-9a-zA-Z]+', '', feature[1]))
        arff.append('@ATTRIBUTE {} NUMERIC'.format(attr_val))
    arff.append('@ATTRIBUTE class {{{}}}\n'.format(','.join(labels_dict.values())))
    arff.append("@DATA")

    for f, l in zip(instances, labels):
        f_list = f.split()
        instance = ""
        for i in range(len(features)):
            instance += str(i) + ' ' + str(float(f_list[i])) + ','
        arff.append("{{{}{} '{}'}}".format(instance, str(len(features)), labels_dict[l]))

    for line in arff:
        print(line)

