import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd
import openpyxl

# reading in our data as well as replacing instances of '?' with a very large value (-9999) as SVM, unlike something like a linear regression isn't sensitive to outliers. Also removing 'ID' column as it serves no predictive power.
df = pd.read_csv('breastCancerdata.data')
df.replace('?', -9999, inplace=True)
df.drop(columns=['id'], inplace=True)

# We can produce a correlation matrix to gauge correlation (Not causation) between independent variables. Relationships greater than 0.70 and less than -0.70 are to be considered highly correlated.
correlation_matrix = df.corr(numeric_only=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# exporting the produced correlation matrix to excel
correlation_matrix.to_excel('correlation_matrix', index=True, engine='openpyxl')

accuracies = []

for i in range(10):
    X = np.array(df.drop(columns=['class'])) # features include everything but label 'class'
    y = np.array(df['class']) # class is everything but features...

    # 80/20 split on both training an test datasets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # selecting the SVM classifier, more specifically the Support Vector Classification as we're classifying here. Kernel set to 'linear' given the linear nature of the data.
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    accuracies.append(accuracy)

print(f'\nThe averaged accuracy was: {sum(accuracies)/len(accuracies)}')

