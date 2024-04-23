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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[13]:


X = df_tracks[['popularity','duration']]
y = df_tracks['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:


#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[16]:


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=40)


# In[17]:


# Train the classifier
rf_classifier.fit(X_train_scaled, y_train)


# In[18]:


y_pred = rf_classifier.predict(X_test_scaled)


# In[19]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:





# In[20]:


df_genre=pd.read_csv('SpotifyFeatures.csv')
df_genre.head(6)


# In[21]:


plt.title('Duration of the Songs in Various Genres')
sns.color_palette('rocket',as_cmap=True)
sns.barplot(y='genre',x='duration_ms',data=df_genre)
plt.xlabel('Duration in Milli Seconds')
plt.ylabel('Genres')


# In[22]:


sns.set_style(style='darkgrid')
plt.figure(figsize=(7,3))
famous=df_genre.sort_values('popularity',ascending=False).head(10)
sns.barplot(y='genre',x='popularity',data=famous).set(title='Top 5 Genres by Popularity')


# In[ ]:




