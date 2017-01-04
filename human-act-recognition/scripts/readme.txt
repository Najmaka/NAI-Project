in order to create arff file provide paths to the features, activity_labels, train_features and train_labels
the order is important

example shell cmd:
./arff_maker.py ../data/features.txt ../data/activity_labels.txt ../data/train_features.txt ../data/train_labels.txt

to save as arff file:
./arff_maker.py ../data/features.txt ../data/activity_labels.txt ../data/train_features.txt ../data/train_labels.txt > dataset.arff
