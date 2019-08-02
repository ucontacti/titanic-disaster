import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    dt = np.dtype([('PassengerId', np.int), ('Pclass', np.int), ('Name', np.str_, 16), ('Sex', np.str_, 16), 
    ('Age', np.float), ('SibSp', np.int), ('Parch', np.int), ('Ticket', np.int),
    ('Fare', np.float), ('Cabin', np.str_, 16), ('Embarked', np.str_, 16)])

    data_val  = pd.read_csv('data/test.csv')
    data = data_val.values
    print(data)
        
    # read information into 1D arrays
    pId = data[:,0].astype(np.int)
    print(pId)