import copy

import pandas as pd
import numpy as np
import graphviz as gv
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
##Task 1 a) The continuous attributes are Name, Age, Fare, Ticket
## The columns missing values are: Cabin
#Using "information gain" as the importance function"

df = pd.read_csv("train.csv")
varen = df["Ticket"]


varik = 0
#df2 = df.loc[df["Sex"] == "male"]
#print(" HERE \n", df.groupby("Survived")["Sex"].value_counts().unstack(fill_value=0).stack())
