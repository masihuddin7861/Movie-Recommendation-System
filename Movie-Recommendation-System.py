#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv.zip')
credits = pd.read_csv('tmdb_5000_credits.csv.zip')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits,on='title')


# In[6]:


credits.shape


# In[7]:


movies.head(1)


# In[8]:


movies.shape


# In[9]:


credits.shape


# In[10]:


movies.head(1)


# In[11]:


# genres
# id
# keyword
#title
# overview
# crew
# cast
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[12]:


movies.head(2)


# # data prerocecing EDA

# In[13]:


movies.info()


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.duplicated().sum()


# In[17]:


movies.iloc[0].genres


# In[18]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','FFantasy','Scifi']


# In[19]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[ ]:





# In[20]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[21]:


movies['genres']=movies['genres'].apply(convert)


# In[22]:


movies.head()


# In[23]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[24]:


movies.head()


# In[25]:


movies['cast'][0]


# In[26]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter +=1
        else:
            break
        
    return L    


# In[27]:


movies['cast']=movies['cast'].apply(convert3)


# In[28]:


movies.head()


# In[29]:


movies['crew'][0]


# In[30]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
        
    return L    


# In[31]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[32]:


movies.head()


# In[33]:


movies['overview'][0]


# In[34]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[35]:


movies.head()


# # apply tranformation

# In[36]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[37]:


movies.head()


# In[38]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[39]:


movies.head()


# In[40]:


new_df = movies[['movie_id','title','tags']]


# In[41]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[42]:


new_df.head()


# In[43]:


new_df['tags'][0]


# In[44]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[45]:


new_df.head()


# # text vectarization

# In[46]:


import nltk


# In[47]:


from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()


# In[48]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)    


# In[49]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[50]:


new_df['tags'][0]


# In[51]:


new_df['tags'][1]


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words = 'english')


# In[53]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[54]:


vectors


# In[55]:


vectors[0]


# In[56]:


cv.get_feature_names()


# In[57]:


from sklearn.metrics.pairwise import cosine_similarity


# In[58]:


similarity = cosine_similarity(vectors)


# In[59]:


similarity[0]


# In[60]:


new_df[new_df['title'] == 'Avatar'].index[0]


# In[61]:


sorted(list(enumerate(similarity[0])),reverse = True,key=lambda x:x[1])[1:6]


# In[62]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[63]:


recommend('Avatar')


# In[64]:


import pickle


# In[65]:


new_df['title'].values


# In[69]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[70]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:


new_df


# In[ ]:


new_df.to_dict()


# In[ ]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




