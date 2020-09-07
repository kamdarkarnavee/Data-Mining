import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def preprocessing(data):
    data.dropna(inplace=True)
    feature_columns = data.columns[data.dtypes == object].values
    for i in feature_columns:
        ch = data.loc[:, i].unique()
        f = [j for j in range(len(data.loc[:, i].unique()))]
        data.loc[:, i] = data.loc[:, i].replace(ch, f)
    return data


def feature_selection():
    correlation_matrix = data.corr()
    print(correlation_matrix)
    top_corr_features = correlation_matrix.index
    plt.figure(figsize=(20, 20))
    sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")  # plot heat map
    plt.show()
    cor_target = abs(correlation_matrix['income'])  # correlation with output variable
    return pd.Series(cor_target).nlargest(n=6).index.tolist()  # returning highly correlated features


def data_selection(data, relevant_features):
    del_cols = data.columns.tolist()
    for i in relevant_features:
        del_cols.remove(i)
    data.drop(columns=del_cols, inplace=True)
    return data


def evaluation_metrics(y_test, y_pred):
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))


def logistic_regression(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test) * 100))
    evaluation_metrics(y_test, y_pred)


def svm_classifier(X_train, X_test, y_train, y_test):
    svc = SVC(kernel='linear', random_state=100)
    svc.fit(X_train, y_train)
    y_predicted = svc.predict(X_test)
    print('Accuracy of SVM classifier on test set:', round(svc.score(X_test, y_test) * 100, 2))
    evaluation_metrics(y_test, y_predicted)


if __name__ == '__main__':
    data = pd.read_csv('adult_census_data.csv')

    print('Initial dataset:')
    print(data.head())

    # Pre-processing and Transformation
    data = preprocessing(data)
    print('After pre-processing the dataset:')
    print(data.head())

    # Feature Selection using correlation matrix
    relevant_features = feature_selection()
    print('Relevant Features:', relevant_features)

    # Data Selection
    data = data_selection(data, relevant_features)
    # data.to_csv('preprocessed_adult_census_data.csv')

    # Splitting data into training and test data sets
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Logistic Regression
    logistic_regression(X_train, X_test, y_train, y_test)

    # SVM (Takes long before displaying result due to heavy computation)
    svm_classifier(X_train, X_test, y_train, y_test)
