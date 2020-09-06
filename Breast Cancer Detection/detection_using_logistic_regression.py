import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    data = pd.read_csv('breastcancer.csv', usecols=['Cl.thickness', 'Cell.size', 'Cell.shape', 'Class'])
    print('Total No. of records:', len(data))
    print(data.head())
    print(data['Class'].value_counts())
    data['Class'] = data['Class'].map({'malignant': 1, 'benign': 0})
    print('Denoting malignant as 1 and benign as 0')
    print(data['Class'].value_counts())
    X = data.loc[:, data.columns != 'Class']
    y = data.loc[:, data.columns == 'Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    train_data = pd.DataFrame(X_train)
    train_data['Class'] = y_train
    data_majority = train_data[train_data.Class == 0]
    data_minority = train_data[train_data.Class == 1]

    # Downsample majority class
    data_majority_downsampled = resample(data_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(data_minority),  # to match minority class
                                       random_state=100)  # reproducible results

    downsampled = pd.concat([data_majority_downsampled, data_minority])
    # downsampled.to_csv('downsampled.csv', index=False)

    # Predicting using Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(downsampled[['Cl.thickness', 'Cell.size', 'Cell.shape']], downsampled['Class'])
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)*100))