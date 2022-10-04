#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install --upgrade tensorflow-hub')
get_ipython().system('pip install gensim')


# In[ ]:


# In[ ]:




# In[20]:


import json
import pandas as pd
import tensorflow_hub as hub
import gensim
from sklearn.preprocessing import MinMaxScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# In[21]:


bio = pd.read_csv('instagram_bio.csv')
cap = pd.read_csv('captions.csv')
score = pd.read_csv('scores.csv')


# In[22]:


score


# In[23]:



# In[24]:


data = pd.concat([bio,cap['caption'],score['follower_count'], score['following_count'],score['tiktok_username'],score['instagram_username']], axis=1)
data['tiktok_username'] = data['tiktok_username'].fillna(0)
data['instagram_username'] = data['instagram_username'].fillna(0)
data['tiktok_username'][data['tiktok_username'] != 0] = 1
data['instagram_username'][data['instagram_username'] != 0] = 1
data['follower_count'][data['follower_count'] == 'None'] = 'nan'
data['following_count'][data['following_count'] == 'None'] = 'nan'
data = data.dropna()


# In[25]:


data.shape


# In[26]:


model = Doc2Vec(vector_size=32, min_count=1, epochs = 20)


# In[27]:


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


# In[9]:


embeddings_bio = embed(data['bio'])
#embeddings_caption = embed(data['caption'])


# In[40]:


data['caption'][3]


# In[41]:


embeddings_caption = embed(data['caption'][3])


# In[246]:


data.to_csv('dat.csv')


# In[247]:


embeddings_bio


# In[248]:


from sklearn.decomposition import PCA
bios= pd.DataFrame(embeddings_bio.numpy())
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(bios)


# In[249]:


from matplotlib import pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = [20.00, 15.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principalComponents[:,0], principalComponents[:,1], principalComponents[:,2], alpha=1, linewidths=5)
plt.legend(["Biographies"],prop={'size': 30})
plt.show()


# In[250]:



data['bio_1'] = np.array(principalComponents[:,0])
data['bio_2'] = np.array(principalComponents[:,1])
data['bio_3'] = np.array(principalComponents[:,2])


# In[251]:


data


# In[252]:


dat = data[['follower_count','following_count', 'tiktok_username', 'instagram_username', 'bio_1','bio_2', 'bio_3','scores']]


# In[211]:


dat = dat.dropna()


# In[253]:


dat.dtypes


# In[254]:


dat.shape


# In[255]:


dat['follower_count'] = dat['follower_count'].astype(str).astype(float)
dat['following_count'] = dat['following_count'].astype(str).astype(float)


# In[256]:


scaler = MinMaxScaler()
dat[['follower_count', 'following_count', 'bio_1', 'bio_2', 'bio_3']] = scaler.fit_transform(dat[['follower_count', 'following_count', 'bio_1', 'bio_2', 'bio_3']])


# In[266]:


dat = dat.dropna()
y=dat['scores']
dat['follower_count'] = dat['follower_count']*6
dat['follower_count'][dat['follower_count'] > 1] = 1

x=dat[['follower_count', 'following_count', 'bio_1', 'bio_2', 'bio_3']]


# In[267]:


x


# In[268]:


x.plot()


# In[269]:


import seaborn as sns


# In[270]:


sns.pairplot(dat[['follower_count', 'following_count', 'bio_1', 'bio_2', 'bio_3']], diag_kind='kde')


# In[271]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.constraints import MaxNorm


# In[301]:


model = Sequential()
model.add(Dense(12, input_shape=(5,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])


# In[302]:


x.isnull().values.any()


# In[303]:


from sklearn.model_selection import train_test_split


# In[304]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[305]:


X_train


# In[306]:


model.fit(X_train,y_train,epochs=500, batch_size=10, verbose=2)


# In[307]:


y_pred = model.predict(X_test)
y_test-list(y_pred[:,0])


# In[308]:


list(y_pred[:,0])


# In[309]:


y_test


# In[310]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test, y_pred))


# In[311]:


from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test, y_pred)


# In[ ]:




