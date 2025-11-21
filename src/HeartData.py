import pandas
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

data = pandas.read_csv('./data/HeartData.csv') 

print(data)

# trainData, testData = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=222)
# trainData, validationData = sklearn.model_selection.train_test_split(trainData, test_size=0.2, random_state=222)
trainData, validationData = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=222)

#model = svm.SVC()
#model = RandomForestClassifier(n_estimators=100)
#model = LinearRegression()
model = LogisticRegression()

trainDataWithoutTarget = trainData.copy()
trainDataWithoutTarget = trainDataWithoutTarget.drop(columns=['target'])

validationDataWithoutTarget = validationData.copy()
validationDataWithoutTarget = validationDataWithoutTarget.iloc[:, :-1] # drop 'target' column, which is last one

model.fit(trainDataWithoutTarget, trainData['target'])
result = model.score(validationDataWithoutTarget, validationData['target'])
print(result)

_, predictData = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=222)
predicted = model.predict(predictData.iloc[:, :-1])
print(predicted)

params = model.get_params()
print(params)

print(model.feature_names_in_)
print(model.n_features_in_)

# I expected I could recreate the model from parameters only, but it seems not to work that way.
# I don't want to train again, just use the saved parameters.
# modelFromParams = LogisticRegression()
# modelFromParams.set_params(**params)
# modelFromParams.predict(predictData.iloc[:, :-1])