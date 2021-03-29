import numpy as np
import pandas as pd

def B(p):
    if p<0:
        raise ValueError("MORDI")
    elif p==1:
        return 0
    elif p==0:
        return 1
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

df = pd.read_csv("train.csv")
varen = df.sort_values(by=["Age"])
nmg = 0