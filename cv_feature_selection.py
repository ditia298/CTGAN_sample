import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  f_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import warnings


warnings.filterwarnings("ignore")

kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)


data = pd.read_csv("trainV1.csv")


data = data.drop("ID_code", axis = 1)
data = data.sample(frac = 0.1)

X = data.iloc[:,0:200]
y = data.iloc[:,-1]


k = list(range(1,21))
for i in range(len(k)):
    k[i] = k[i]*10

print(k)

for i in k:
    bestfeatures = SelectKBest(score_func=f_classif, k=i)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns


    l=[]
    for x in featureScores.nlargest(i,'Score').Specs:
      l.append(x)
    train_data = X[l]

    model = RandomForestClassifier(max_depth=20, max_features=i)


    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    scores = cross_val_score(model, train_data, y, cv = cv)
    print(scores)
    print("Accuracy for" + str(i) + "features: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("")
