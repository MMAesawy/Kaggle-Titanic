# coding: utf-8
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import random
import os
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

random_seed = 3398655969

def main():
    df_train = pd.read_csv('train.csv')

    # Dropping unneeded columns
    df_train = df_train.drop(['PassengerId', 'Cabin', 'Ticket'], axis = 1)

    # Processing the 'Name' column.
    p = re.compile(r"\b[a-z]*[.]\s+", re.IGNORECASE)
    name_map = {'mrs.': 0, 'mr.': 0, 'mlle.': 0,
                             'mme.': 0, 'ms.': 0, 'miss.': 0,
                             'master.': 0, 'dr.': 1, 'rev.': 1,
                             'major.': 2, 'don.': 1, 'dona.': 1,
                             'countess.': 2, 'lady.': 2, 'sir.': 2,
                             'col.': 2, 'capt.': 2, 'jonkheer.': 2}
    df_train['Name'] = df_train['Name'].apply(lambda name: p.search(name).group(0).strip())
    df_train['Name'] = df_train['Name'].apply(lambda name: name_map[name.lower()])

    # Processing the 'Age' column
    df_train['Age'].fillna(value = df_train['Age'].median(skipna = True), inplace = True)
    age_se = StandardScaler()
    df_train['Age'] = age_se.fit_transform(df_train['Age'].values.reshape(-1,1))

    # Processing the 'Fare' column
    fare_se = StandardScaler()
    df_train['Fare'] = fare_se.fit_transform(df_train['Fare'].values.reshape(-1,1))

    # Processing the 'Parch' column
    parch_se = StandardScaler()
    df_train['Parch'] = parch_se.fit_transform(df_train['Parch'].values.reshape(-1,1))

    # Processing the "SibSp' column
    sibsp_se = StandardScaler()
    df_train['SibSp'] = sibsp_se.fit_transform(df_train['SibSp'].values.reshape(-1,1))

    # Processing the 'Sex' column
    sex_encoder = LabelEncoder()
    df_train['Sex'] = sex_encoder.fit_transform(df_train['Sex'])

    # Processing the 'Embarked' column
    embarked_encoder = LabelEncoder()
    df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
    df_train['Embarked'] = embarked_encoder.fit_transform(df_train['Embarked'])
    embarked_ohe = OneHotEncoder(sparse = False)
    ohe_encoding = embarked_ohe.fit_transform(df_train['Embarked'].values.reshape(-1, 1))
    df_train = df_train.drop(['Embarked'], axis = 1)
    df_train = pd.concat([df_train, pd.DataFrame(ohe_encoding[:, :-1], columns = ['Embarked0', 'Embarked1'])], axis = 1)

    df_train['Sex'] = np.where(df_train['Sex'] == 0, -1, 1)

    return df_train

def fit(df_train):
    X_all = df_train.iloc[:, 1:].values
    print(X_all.shape)
    print(df_train.shape)
    y = df_train['Survived'].values


    #for i in range(X_all.shape[1]):
       # print('\n\n\n------- CURRENT VARIABLE: ', df_train.columns[i + 1], ' -------\n\n\n')
       # X = X_all[:, i].reshape(-1, 1)
    X = X_all
    print("Grid searching LDA model... ")
    lda_gs = GridSearchCV(cv=10, n_jobs=5, scoring='accuracy', verbose=1,
                          estimator=LinearDiscriminantAnalysis(),
                          param_grid=[{'solver': ['lsqr'],
                                       'shrinkage': [x / 100 for x in range(0, 100, 1)]}])
    lda_gs.fit(X, y)
    print(f'Success! Best score is {lda_gs.best_score_}')
    print(lda_gs.best_params_)
    print("Grid searching RFC model... ")
    rfc_gs = GridSearchCV(cv=10, n_jobs=5, scoring='accuracy', verbose=1,
                          estimator=RandomForestClassifier(n_jobs=-1,
                                                           max_features='sqrt',
                                                           min_samples_split=6,
                                                           random_state=random_seed),
                          param_grid=[{'max_depth': [x for x in range(1, 100)]}])
    rfc_gs.fit(X, y)
    print(f'Success! Best score is {rfc_gs.best_score_}')
    print(rfc_gs.best_params_)
    print("Grid searching KNN model... ")
    knn_gs = GridSearchCV(cv=10, n_jobs=5, scoring='accuracy', verbose=1,
                          estimator=KNeighborsClassifier(),
                          param_grid=[{'n_neighbors': [x for x in range(3, 100)],
                                       'weights': ['uniform', 'distance']}])
    knn_gs.fit(X, y)
    print(f'Success! Best score is {knn_gs.best_score_}')
    print(knn_gs.best_params_)
    print("Grid searching Log model... ")
    log_gs = GridSearchCV(cv=10, n_jobs=5, scoring='accuracy', verbose=1,
                          estimator=LogisticRegression(random_state=random_seed, n_jobs=-1),
                          param_grid=[{'penalty': ['l1', 'l2'],
                                       'C': [x ** 2 / 1000 for x in range(1, 400, 5)]}])
    log_gs.fit(X, y)
    print(f'Success! Best score is {log_gs.best_score_}')
    print(log_gs.best_params_)
    print("Grid searching SVM model... ")
    svm_gs = GridSearchCV(cv=10, n_jobs=5, scoring='accuracy', verbose=1,
                          estimator=SVC(max_iter=6000),
                          param_grid=[{'C': [x ** 2 / 1000 for x in range(1, 200, 10)],
                                       'kernel': ['linear', 'poly', 'sigmoid']}
                              , {'C': [x ** 2 / 1000 for x in range(1, 200, 10)],
                                 'kernel': ['rbf'],
                                 'gamma': [x ** 2 / 2000 for x in range(1, 30, 1)]}])
    svm_gs.fit(X, y)
    print(f'Success! Best score is {svm_gs.best_score_}')
    print(svm_gs.best_params_)
    print()
    print(f'SVM score: {svm_gs.best_score_:.4}')
    print(f'LDA score: {lda_gs.best_score_:.4}')
    print(f'Random forests score: {rfc_gs.best_score_:.4}')
    print(f'KNN score: {knn_gs.best_score_:.4}')
    print(f'Logistic Reg. score: {log_gs.best_score_:.4}')
    print()
    print(svm_gs.best_params_)
    print(lda_gs.best_params_)
    print(rfc_gs.best_params_)
    print(knn_gs.best_params_)
    print(log_gs.best_params_)

if __name__ == '__main__':
    df = main()
    #fit(df)

    vc = VotingClassifier(estimators=[
        ('svm', SVC(C=14.641, gamma = 0.032, kernel = 'rbf', probability=True)),
        ('lda', LinearDiscriminantAnalysis(shrinkage=0.26, solver='lsqr')),
        ('randomforest', RandomForestClassifier(max_depth=11, min_samples_split=6, max_features='sqrt')),
        ('log', LogisticRegression(C = 0.032, penalty='l2')),
        ('knn', KNeighborsClassifier(n_neighbors=29, weights='uniform'))
    ], voting='soft')
    cv = StratifiedKFold(10, random_state=random_seed)
    X = df.iloc[:, 1:].values
    y = df['Survived'].values
    cv_errors = []
    for train, test in cv.split(X, y):
        X_train, X_test = X[train,:], X[test,]
        y_train, y_test = y[train], y[test]
        vc.fit(X_train, y_train)
        y_pred = vc.predict(X_test)
        cv_errors.append(accuracy_score(y_test, y_pred))
    print(sum(cv_errors)/len(cv_errors))



    #os.system('rundll32.exe PowrProf.dll,SetSuspendState 0,1,0')


'''
LDA score: 0.7924
Random forests score: 0.8384
KNN score: 0.8148
Logistic Reg. score: 0.7969

{'shrinkage': 0.26, 'solver': 'lsqr'}
{'max_depth': 11}
{'n_neighbors': 29, 'weights': 'uniform'}
{'C': 0.036, 'penalty': 'l2'}
'''