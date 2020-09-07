# k-nearest neighbors on the Adult Census Dataset
from random import seed
from random import randrange
from math import sqrt
import pandas as pd


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = dataset.values.tolist()
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args, distance_metrics):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args, distance_metrics)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Calculate the Manhattan distance between two vectors
def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i])
    return distance


# Calculate the Chebyshev distance between two vectors
def chebyshev_distance(row1, row2):
    distance = []
    for i in range(len(row1) - 1):
        distance.append(abs(row1[i] - row2[i]))
    return max(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, distance_metrics):
    distances = list()
    for train_row in train:
        dist = distance_metrics(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors, distance_metrics):
    neighbors = get_neighbors(train, test_row, num_neighbors, distance_metrics)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors, distance_metrics):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors, distance_metrics)
        predictions.append(output)
    return predictions


# Test the kNN on Adult Census dataset
seed(1)
filename = 'preprocessed_adult_census_data.csv'
dataset = pd.read_csv(filename, header=None, index_col=0)
dataset = dataset.sample(n=1000, random_state=1)
dataset = dataset.apply(pd.to_numeric)
# # evaluate algorithm
n_folds = 5
num_neighbors = 5
distance_measures = [euclidean_distance, manhattan_distance, chebyshev_distance]
for i in distance_measures:
    scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors, distance_metrics=i)
    print('Scores: %s' % scores)
    print('Mean Accuracy of %s: %.3f%%' % (i.__name__, sum(scores) / float(len(scores))))
