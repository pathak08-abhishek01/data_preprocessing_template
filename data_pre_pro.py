import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def prepro(fileloc, m):
    # Reading Data Set from CSV file named Data
    dataset = pd.read_csv(fileloc)

    # Creating Matrix of features
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values

    # Taking care of Missing Data
    imputer = SimpleImputer(missing_values=np.nan, strategy=m)
    imputer = imputer.fit(x[:, 1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])

    # Categorical Encoding and Dummy Encoding
    labelencoder_x = LabelEncoder()
    x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
    onehotencoder = OneHotEncoder(categorical_features=[0])
    x = onehotencoder.fit_transform(x).toarray()
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Splitting Data Set in test and train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Returning Values
    return x_train, x_test, y_train, y_test