import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn as skl 
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error


random.seed(0)

data = pd.read_csv("data/afbrs_transdf.csv")
data = data.drop(columns=['X'])
#data = data.melt(id_vars=['priority', 'expectation', 'region'], value_vars=['choice_one', 'choice_two', 'choice_three'])
#print(data.head())

encoder = LabelEncoder()
one_hot_encoding = OneHotEncoder()
for i in data.columns:
    data[i]= encoder.fit_transform(data[i])

data_features = data.drop(columns=['choice'])
data_labels = data['choice']

X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.3, random_state=0)

topics_list = list(range(6))

# Fit estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesClassifier(n_estimators=20, random_state=0),
    "Support Vector Machine": LinearSVC(max_iter=1000),
    "Logistic Regression": LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500),
    "Perceptron": MLPClassifier(hidden_layer_sizes=(50,), batch_size=50, max_iter=500),
    "MUlti Layer Perceptron": MLPClassifier(hidden_layer_sizes=(300,)),
    "linear discriminant Analysis" : LinearDiscriminantAnalysis(),
    "Decision Tree": DecisionTreeClassifier()
}

y_test_predict = dict()
output_results = []
classifiers = []
results_dic = {}
for name, estimator in ESTIMATORS.items():
    result = {}
    estimator.fit(X_train, y_train)
    predicted_result = estimator.predict(X_test)
    y_test_predict[name] = predicted_result
    prediction_score = accuracy_score(y_test.to_numpy(), y_test_predict[name])
    regions_test = X_test['region'].to_numpy()
    df = pd.DataFrame(np.stack((regions_test, predicted_result), axis=1))
    df = df.groupby(df[0])
    for v in df:
        choices = list(v[1][1])
        region_index = v[1][0].iloc[0]
        choice_priority = max(choices,key=choices.count)
    result['precision'] = precision_score(y_test_predict[name], y_test.to_numpy(), labels=[1, 2, 3], average="micro")
    result['recall'] = recall_score(y_test_predict[name], y_test.to_numpy(),  labels=[1, 2, 3], average="micro")
    result['f1Score'] = f1_score(y_test_predict[name], y_test.to_numpy(), labels=[1, 2, 3], average="micro")
    result['accuracy'] = prediction_score
    result['MSE'] = mean_squared_error(y_test_predict[name], y_test.to_numpy())
    classifiers.append(name)
    output_results.append(round(prediction_score*100, 3))
    results_dic[name] = result

print('The predicted metrics', results_dic)

plt.rcdefaults()
fig, ax = plt.subplots()
print(classifiers)
print(output_results)
# Example data
y_pos = np.arange(len(classifiers))
performance = output_results

ax.barh(y_pos, performance)
ax.set_yticks(y_pos)
ax.set_yticklabels(classifiers)
ax.invert_yaxis() 
ax.set_xlabel('Accuracy of Prediction %')
ax.set_title('Prediction of priority choice')

plt.show()
