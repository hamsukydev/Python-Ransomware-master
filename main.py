import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv',encoding='latin-1')
print(df.head())

## Now let’s start working on the dataset and make some amazing visualisations and conclusions.

##Data Cleaning

print(df.info())


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace= True)

df.rename(columns={'v1':'message_type', 'v2':'message'},inplace=True)
print(df.sample(5))

# ## As this is a classification problem we want the “message_type” to be binary classified i.e, 0 or 1 so for this
# purpose we use label encoder.

from sklearn.preprocessing import LabelEncoder
encoder =LabelEncoder()

df['message_type']=encoder.fit_transform(df['message_type'])
df['message_type'].sample(5)

print(df[df['message_type']==1])


## Now let’s check for the missing values

print(df.isnull().sum())
print(df.duplicated().sum())

## There are 403 duplicated values and we have to remove them

df= df.drop_duplicates()

## Exploratory Data Analysis
## Let’s visualise the classification problem to get a better understanding of the data.



df['message_type'].value_counts()
plt.pie(df['message_type'].value_counts(),labels=[' not spam','spam'],autopct='%0.2f')
plt.show()