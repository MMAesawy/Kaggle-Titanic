# coding: utf-8
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Imputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from collections import defaultdict
random_seed = 3398655969
df_train = pd.read_csv('train.csv')

# Dropping unneeded columns
df_train = df_train.drop(['PassengerId', 'Cabin', 'Ticket'], axis = 1)

# Processing the 'Name' column.
p = re.compile(r"\b[a-z]*[.]\s+", re.IGNORECASE)
name_map = defaultdict(int)
name_map.update({'mrs.': 0, 'mr.': 0, 'mlle.': 0,
                         'mme.': 0, 'ms.': 0, 'miss.': 0,
                         'master.': 0, 'dr.': 1, 'rev.': 1,
                         'major.': 2, 'don.': 1, 'dona.': 1,
                         'countess.': 2, 'lady.': 2, 'sir.': 2,
                         'col.': 2, 'capt.': 2, 'jonkheer.': 2})
df_train['Name'] = df_train['Name'].apply(lambda name: p.search(name).group(0).strip())
df_train['Name'] = df_train['Name'].apply(lambda name: name_map[name.lower()])

# Processing the 'Age' column
age_median = df_train['Age'].median(skipna = True)
df_train['Age'].fillna(value = age_median, inplace = True)
age_se = StandardScaler()
df_train['Age'] = age_se.fit_transform(df_train['Age'].values.reshape(-1,1))

# Processing the 'Fare' column
fare_median = df_train['Fare'].median(skipna = True)
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
df_train = pd.concat([df_train, pd.Series(df_train['Sex'] * df_train['Pclass'])], axis = 1)
X = df_train.iloc[:, 1:].values
y = df_train['Survived'].values
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10,max_features=None,min_samples_split=8,max_depth = 18,random_state=random_seed)
rfc.fit(X, y)
df_test = pd.read_csv('test.csv')
df_test.drop(['Cabin', 'Ticket'], axis = 1, inplace = True)
df_test['Name'] = df_test['Name'].apply(lambda name: p.search(name).group(0).strip())
df_test['Name'] = df_test['Name'].apply(lambda name: name_map[name.lower()])
df_test['Age'].fillna(value = age_median, inplace = True)
df_test['Age'] = age_se.transform(df_test['Age'].values.reshape(-1, 1))
df_test['Fare'].fillna(value = fare_median, inplace = True)
df_test['Fare'] = fare_se.transform(df_test['Fare'].values.reshape(-1, 1))
df_test['Parch'] = parch_se.transform(df_test['Parch'].values.reshape(-1,1))
df_test['SibSp'] = sibsp_se.transform(df_test['SibSp'].values.reshape(-1,1))
df_test['Sex'] = sex_encoder.transform(df_test['Sex'])
df_test['Sex'] = np.where(df_test['Sex'] == 0, -1, 1)
df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Embarked'] = embarked_encoder.transform(df_test['Embarked'])
ohe_encoding = embarked_ohe.transform(df_test['Embarked'].values.reshape(-1, 1))
df_test = df_test.drop(['Embarked'], axis = 1)
df_test = pd.concat([df_test, pd.DataFrame(ohe_encoding[:, :-1], columns = ['Embarked0', 'Embarked1'])], axis = 1)
df_test = pd.concat([df_test, pd.Series(df_test['Sex'] * df_test['Pclass'])], axis = 1)
y_pred = rfc.predict(df_test.values[:, 1:])
out = pd.concat([df_test['PassengerId'], pd.DataFrame(y_pred, columns = ['Survived'])], axis = 1)
out.to_csv('preds.csv')
