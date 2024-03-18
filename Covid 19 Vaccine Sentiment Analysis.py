#!/usr/bin/env python
# coding: utf-8

# In[9]:


#loading necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


data=pd.read_csv(r"C:\Users\KIIT\Downloads\Covid Vaccine Tweets\vaccination_all_tweets.csv")
data.head()


# In[11]:


data.drop(columns=['id', 'user_created', 'user_friends', 'retweets', 'favorites', 'is_retweet'], inplace=True)
data.info()


# In[16]:


data=data.dropna()
data.info()


# In[19]:


unique_locations_count = data['user_location'].nunique()
print(unique_locations_count)


# In[20]:


locations_counts = data['user_location'].value_counts()
print(locations_counts)


# In[22]:


filtered_locations_counts = locations_counts[locations_counts > 1000]
filtered_locations_counts.plot(kind='bar')
plt.show()


# In[23]:


# Create a mask for rows where 'user_location' contains 'India'
mask = data['user_location'].str.contains('India', na=False)

# Assign 'India' to all rows where the mask is True
data.loc[mask, 'user_location'] = 'India'
# Create a new column 'country' based on the 'user_location' column
data['country'] = data['user_location'].where(~mask, other='India')


# In[29]:


country_counts = data['country'].value_counts()
print(country_counts)


# In[30]:


filtered_country_counts = country_counts[country_counts > 500]
filtered_country_counts.plot(kind='bar')
plt.show()


# In[36]:


users_count=data['user_verified'].value_counts()


# In[39]:


users_count.plot(kind='bar')
plt.xlabel('Users')
plt.ylabel('Count of Verified users')
plt.title('Count of Verified users and Unverified users')


# In[42]:


data.drop(columns=['user_followers','user_favourites','user_location','source','hashtags'],inplace=True)
data.info()


# In[43]:


data['date'] = pd.to_datetime(data['date'])


# In[44]:


data.head()


# In[47]:


# Remove opening and closing brackets
data.text = data.text.str.strip("[']")
# remove all quotes too
data.text = data.text.str.replace(" ', '", ",", regex = False)


# In[49]:


import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[53]:


import time
import nltk

# Download stopwords resource
nltk.download('stopwords')

# Rest of your code
start = time.time()
cache = set(stopwords.words("english"))

def remove_stopwords(words):
    new_text = " ".join([word for word in words.split() if word not in cache])
    return new_text

data.text=data.text.apply(remove_stopwords)


# In[62]:


data.text = data.text.str.strip("#")
data['text'] = data['text'].str.replace(r'#\S+', '', regex=True)


# In[63]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a text:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(words):    
    if words == "No Negative" or words == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(words)["compound"]


# In[76]:


start=time.time()
data["Sentiment_score"] = data.text.apply(calc_sentiment)
end=time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")


# In[78]:


def sentiment_score(score):
    if score > 0.6:
        return "It's a positive comment"
    else:
        return "It's a negative comment"
data['Sentiment_score'].apply(sentiment_score)


# In[79]:


import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame and it has a 'sentiment_score' column
# Categorize sentiments
data['Sentiment_Category'] = data['Sentiment_score'].apply(lambda x: 'Positive' if x > 0.6 else 'Negative')

# Count the occurrences of each category
sentiment_counts = data['Sentiment_Category'].value_counts()

# Plot the counts
sentiment_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Sentiment Scores')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# In[80]:


data=data.drop(columns=['user_description','Sentiment'])
data.head()


# In[81]:


data['text']


# In[ ]:




