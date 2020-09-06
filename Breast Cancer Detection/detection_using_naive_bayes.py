import pandas as pd
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")


def calculate_probability(x, mean, stdev):
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))
    return np.prod((1 / (np.sqrt(2 * math.pi) * stdev)) * exponent)


def downsampling(train_data, l, s):
    data_majority = train_data[train_data.Class == l]
    data_minority = train_data[train_data.Class == s]

    # Downsample majority class
    data_majority_downsampled = resample(data_majority,
                                         replace=False,  # sample without replacement
                                         n_samples=len(data_minority),  # to match minority class
                                         random_state=100)  # reproducible results

    return pd.concat([data_majority_downsampled, data_minority])


data = pd.read_csv('breastcancer.csv', usecols=['Cl.thickness', 'Cell.size', 'Cell.shape', 'Class'])
print('Total No. of records:', len(data))
print(data.head())
print(data['Class'].value_counts())
data['Class'] = data['Class'].map({'malignant': 1, 'benign': 0})
print('Denoting malignant as 1 and benign as 0')

X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print('Split Train and Test data')

# Data Pre-processing
train_data = pd.DataFrame(X_train)
train_data['Class'] = y_train
train_data.dropna(inplace=True)

len_class0, len_class1 = train_data['Class'].value_counts()

if len_class0 != len_class1:
    print('Downsampling...')
    if len_class0 > len_class1:
        l, s = 0, 1
    else:
        l, s = 1, 0
    train_data = downsampling(train_data, l, s)

class0 = train_data.iloc[:, :-1][train_data.iloc[:, -1] == 0]
class1 = train_data.iloc[:, :-1][train_data.iloc[:, -1] == 1]

# Data Modelling
class0_mean = class0.apply(np.mean, 0)
class1_mean = class1.apply(np.mean, 0)
class0_std = class0.apply(np.std, 0)
class1_std = class1.apply(np.std, 0)

# Predicting classes
print('Predicting...')
predicted_class = []
for i in range(len(X_test)):
    prob0 = calculate_probability(np.array(X_test.iloc[i, :]), np.array(class0_mean), np.array(class0_std))
    prob1 = calculate_probability(np.array(X_test.iloc[i, :]), np.array(class1_mean), np.array(class1_std))
    predicted_class.append(0 if (prob0 >= prob1) else 1)

# Accuracy
answer = len([i for i, j in zip(predicted_class, y_test['Class']) if i == j])
print('Accuracy of Naive Bayes:', round(answer / len(y_test) * 100, 2), '%')