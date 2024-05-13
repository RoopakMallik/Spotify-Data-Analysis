#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Spotify Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_tracks=pd.read_csv('SpotifyFeatures.csv')


# In[3]:


df_tracks.head(6)


# In[4]:


# checking for null values
pd.isnull(df_tracks)


# In[5]:


df_tracks.info()


# In[6]:


# top 10 least popular songs on Spotify
leastpopular_df_tracks=df_tracks.sort_values('popularity',ascending=True).head(11)
leastpopular_df_tracks


# In[ ]:





# In[7]:


# most popular songs on Spotify
most_popular_tracks=df_tracks.query('popularity>90',inplace=False).sort_values('popularity',ascending=True)
most_popular_tracks[:10]


# In[8]:


# changing time duration of music to seconds
df_tracks['duration']=df_tracks['duration_ms'].apply(lambda x:round(x/1000))
df_tracks.drop('duration_ms',inplace=True,axis=1)


# In[9]:


df_tracks.duration.head(6)


# In[10]:


sample_df=df_tracks.sample(int(0.004*len(df_tracks)))
print(len(df_tracks))


# In[11]:


# Correlation between Loudness and Energy in Music
plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y='loudness',x='energy',color='c').set(title='Loudness vs Energy')


# In[12]:


dataframe = pd.read_csv('SpotifyFeatures.csv')
dataframe.head()


# In[13]:


dataframe.describe()


# In[14]:


print(dataframe.keys())


# In[15]:


list_of_keys = dataframe['key'].unique()
for i in range(len(list_of_keys)):
    dataframe.loc[dataframe['key'] == list_of_keys[i], 'key'] = i
dataframe.sample(5)


# In[16]:


dataframe.loc[dataframe["mode"] == 'Major', "mode"] = 1
dataframe.loc[dataframe["mode"] == 'Minor', "mode"] = 0
dataframe.sample(5)


# In[17]:


list_of_time_signatures = dataframe['time_signature'].unique()
for i in range(len(list_of_time_signatures)):
    dataframe.loc[dataframe['time_signature'] == list_of_time_signatures[i], 'time_signature'] = i
dataframe.sample(5)


# In[18]:


dataframe.loc[dataframe['popularity'] < 57, 'popularity'] = 0 
dataframe.loc[dataframe['popularity'] >= 57, 'popularity'] = 1
dataframe.loc[dataframe['popularity'] == 1]


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[20]:


features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", 
            "mode", "speechiness", "tempo", "time_signature", "valence"]


# In[21]:


training = dataframe.sample(frac = 0.8,random_state = 420)
X_train = training[features]
y_train = training['popularity']
X_test = dataframe.drop(training.index)[features]


# In[22]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 420)


# In[23]:


# Logistic Regression Implementation
LR_Model = LogisticRegression()
LR_Model.fit(X_train, y_train)
LR_Predict = LR_Model.predict(X_valid)
LR_Accuracy = accuracy_score(y_valid, LR_Predict)
print("Accuracy: " + str(LR_Accuracy))

LR_AUC = roc_auc_score(y_valid, LR_Predict) 
print("AUC: " + str(LR_AUC))


# In[24]:


# Random Forest Implementation
RFC_Model = RandomForestClassifier()
RFC_Model.fit(X_train, y_train)
RFC_Predict = RFC_Model.predict(X_valid)
RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
print("Accuracy: " + str(RFC_Accuracy))

RFC_AUC = roc_auc_score(y_valid, RFC_Predict) 
print("AUC: " + str(RFC_AUC))


# In[25]:


df_genre=pd.read_csv('SpotifyFeatures.csv')
df_genre.head(6)


# In[26]:


plt.title('Duration of the Songs in Various Genres')
sns.color_palette('rocket',as_cmap=True)
sns.barplot(y='genre',x='duration_ms',data=df_genre)
plt.xlabel('Duration in Milli Seconds')
plt.ylabel('Genres')


# In[27]:


sns.set_style(style='darkgrid')
plt.figure(figsize=(7,3))
famous=df_genre.sort_values('popularity',ascending=False).head(10)
sns.barplot(y='genre',x='popularity',data=famous).set(title='Top 5 Genres by Popularity')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




