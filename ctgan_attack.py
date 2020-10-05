import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ctgan import CTGANSynthesizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

data = pd.read_csv("trainV1.csv")

data = data.drop("ID_code", axis=1)
data = data.sample(frac=0.2)

X = data.iloc[:, 0:200]
y = data.iloc[:, -1]

def no_feature_selection():

    df = pd.DataFrame({'normal': [], 'poisoned': []})
    l = np.array(["Test1", "Test2", "Test3", "Test4", "Test5", "Test6", "Test7", "Test8", "Test9", "Test10"])
    l = pd.Series(l)
    df['test'] = l
    c = np.array([])
    d = np.array([])

    for i in range(0, 10):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        model = DecisionTreeClassifier(max_features="auto")
        model.fit(X_train, y_train)

        # Predict the response for test dataset
        rf_pred = model.predict(X_test)

        d = np.append(d, metrics.accuracy_score(y_test, rf_pred))

        remove_n = int(len(X_train) * 0.2)

        drop_indices = np.random.choice(X_train.index, remove_n, replace=False)

        X_train["target"] = y_train

        ctgan = CTGANSynthesizer()

        ctgan.fit(X_train, epochs=5)

        samples = ctgan.sample(remove_n)

        # print(samples)
        print("")

        X1 = samples.iloc[:, 0:200]
        y1 = samples.iloc[:, -1]
        y1 = y1.astype(int)

        X_train = X_train.drop("target", axis=1)

        X_train = X_train.drop(drop_indices)
        y_train = y_train.drop(drop_indices)

        X_train = X_train.append(X1)
        y_train = y_train.append(y1)

        # print(X_train)
        # print(y_train)

        model = DecisionTreeClassifier(max_features="auto")
        model.fit(X_train, y_train)

        # Predict the response for test dataset
        rf_pred = model.predict(X_test)

        c = np.append(c, metrics.accuracy_score(y_test, rf_pred))
        # df['poisoned'] = df['poisoned'].metrics.accuracy_score(y_test, rf_pred)

    c = pd.Series(c)
    d = pd.Series(d)
    df['normal'] = d
    df['poisoned'] = c

    df.plot(kind='line', x='test', y=['normal', 'poisoned'])
    plt.savefig('no_feature_selection_70_30.png')

    df.to_csv("no_feature_selection_70_30.csv", index=False)

def feature_selection():

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    df = pd.DataFrame({'normal': [], 'poisoned': []})
    l = np.array(["Test1", "Test2", "Test3", "Test4", "Test5", "Test6", "Test7", "Test8", "Test9", "Test10"])
    l = pd.Series(l)
    df['test'] = l
    c = np.array([])
    d = np.array([])

    for i in range(0,10):
        bestfeatures = SelectKBest(score_func=f_classif, k=160)
        fit = bestfeatures.fit(X,y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']  #naming the dataframe columns


        ll=[]
        for x in featureScores.nlargest(160,'Score').Specs:
          ll.append(x)
        train_data = X[ll]

        #building the tree
        X1_new_train, X1_new_test, y1_new_train, y1_new_test = train_test_split(train_data, y, test_size=0.3, random_state=1)

        model_160 = DecisionTreeClassifier(max_features="auto")


        model_160.fit(X1_new_train, y1_new_train)


        rf_pred = model_160.predict(X1_new_test)

        d = np.append(d, metrics.accuracy_score(y1_new_test, rf_pred))

        remove_n = int(len(X1_new_train)*0.2)

        print(remove_n)

        drop_indices = np.random.choice(X1_new_train.index, remove_n, replace=False)

        ctgan = CTGANSynthesizer()

        X1_new_train["target"] = y1_new_train

        ctgan.fit(X1_new_train, epochs=5)

        # ctgan.save(os.getcwd())

        samples = ctgan.sample(remove_n)

        X1 = samples.iloc[:,0:160]
        y1 = samples.iloc[:,-1]
        y1 = y1.astype(int)


        X1_new_train = X1_new_train.drop("target", axis=1)

        X1_new_train = X1_new_train.drop(drop_indices)
        y1_new_train = y1_new_train.drop(drop_indices)

        X_train = X1_new_train.append(X1)
        y_train = y1_new_train.append(y1)

        model = DecisionTreeClassifier(max_features="auto")
        model.fit(X_train, y_train)

        #Predict the response for test dataset
        rf_pred = model.predict(X1_new_test)

        c = np.append(c, metrics.accuracy_score(y1_new_test, rf_pred))

    c = pd.Series(c)
    d = pd.Series(d)
    df['normal'] = d
    df['poisoned'] = c

    df.plot(kind='line', x='test', y=['normal', 'poisoned'])
    plt.savefig('feature_selection_70_30.png')

    df.to_csv("feature_selection_70_30.csv", index=False)


no_feature_selection()

feature_selection()