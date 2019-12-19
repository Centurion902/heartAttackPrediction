from sklearn.model_selection import train_test_split
import pickle
##read data
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import resample


import pandas as pd
f=pd.read_csv("data/heartAttack.csv")

#add flag for missing values

#tempData = pd.DataFrame(data={'slopeFlag': [], 'caFlag': [], 'thalFlag': []})
#tempData["slopeFlag"] = f["slope"].replace('?', 0)
#tempData["slopeFlag"][f["slope"] != '?'] = 1
#tempData["caFlag"] = f["ca"].replace('?', 0)
#tempData["caFlag"][f["ca"] != '?'] = 1
#tempData["thalFlag"] = f["thal"].replace('?', 0)
#tempData["thalFlag"][f["thal"] != '?'] = 1
#f = pd.concat([tempData, f], axis=1)

##this is commented out because it was found to only have negative or no effect on training through crossvalidation
#if you choose to uncomment this, be sure to add 'thalFlag', 'caFlag', "slopeFlag" to the begining of keepcol

#replace question marks
f["slope"]= f["slope"].replace('?', 0) 
f["ca"]= f["ca"].replace('?', 0) 
f["thal"]= f["thal"].replace('?', 0) 
new_f=f

#keep only nescesary columns
keep_col = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',"slope", "ca", "thal", 'num']
#new_f = f[keep_col]

#removing rows with missing values
for col in keep_col:
    new_f = new_f[new_f[col] != '?']

#do one hot encoding
onehot1 = pd.get_dummies(new_f['restecg'],prefix=['restecg']) ## add, drop_first=True for dummy encoding
onehot2 = pd.get_dummies(new_f['cp'],prefix=['cp']) ## add, drop_first=True for dummy encoding


##dropping old columns converted to one hot.
keep_col.remove('cp')
keep_col.remove('restecg')

new_f = new_f[keep_col]
horizontal_stack = pd.concat([onehot1, onehot2, new_f], axis=1)##concatanatind dataframes

##balancing dataset

# Separate majority and minority classes
df_majority = horizontal_stack[horizontal_stack.num==0]
df_minority = horizontal_stack[horizontal_stack.num==1]
print(df_majority.shape)
print(df_minority.shape)

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print(df_upsampled.shape)

##print(horizontal_stack)
df_upsampled.to_csv("data/heartAttackClean.csv", index=False)


