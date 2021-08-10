# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:29:34 2021

@author: saman
"""
from All_files import *



side_bar=st.beta_container()


with side_bar:
    st.sidebar.image("images/Logo_v.png", width=250,)
    st.sidebar.title("Twitter Analytics option") 
    user_box=st.sidebar.checkbox("User Analysis")
    extract_box = st.sidebar.checkbox("Extract Tweets") 
    analyse_box = st.sidebar.checkbox("Analyse Custom Query")
    if user_box:
        user_box=True
    if user_box:
        menu = ["Stephen King", "Kai", "Sign Up", "Learn"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Stephen King":
                user_df = pd.read_csv("StephenKing.csv" )
                st.write("The intial dataset")
                st.dataframe(user_df)
                # acquires both tweet polarity and subjectivity
                user_df['sentiment'] = user_df['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment)
                # only polarity
                user_df['polarity_score'] = user_df['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               colors=["limegreen", "red", "gray"]) 
                su=plt.axis('equal')
                su=plt.title("Franction of each sentiment in random tweets")
                su=plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left")
                st.write(su.figure)
 
                tweet_dataset = user_df.copy()
                #Removing non-ascii characters (for example, arabian chars)
                user_df['tweet'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
                #Making all fields string type
                for i in range(len(user_df)): 
                    user_df.at[i,'tweet'] = str(user_df.iloc[i]['tweet'])
                for i in range(len(user_df)): 
                    user_df.at[i,'tweet'] =remove_urls(user_df.iloc[i]['tweet'])
                # Convert to list
                datal= user_df['tweet'].values.tolist()
                datal = [re.sub('\S*@\S*\s?', '', sent) for sent in  datal]
                datal = [re.sub('\s+', ' ', sent) for sent in datal]
                datal = [re.sub("\'", "", sent) for sent in datal]
                data_s = np.array(datal)
                tokenizer = Tokenizer(num_words=max_words)
                test_sequence = tokenizer.texts_to_sequences(data_s)
                test_sequence = pad_sequences(test_sequence, maxlen=2500)
                test_prediction = model.predict(test_sequence)
                np.around(test_prediction, decimals=0)
          
                tweet_dataset['label'] = np.around(test_prediction, decimals=0)
                tweet_dataset[tweet_dataset['label']==1.0].head(10) 
              
                for i in range(10):
                    st.write(tweet_dataset.iloc[i*2]['text']) 
                    st.write('\n')
    
    
    if analyse_box:
     st.sidebar.title("Twitter Analysis Input Form")
     dataset_file = st.sidebar.file_uploader(
                    "Upload Tweet Dataset", type=["csv"]
                )
     

    
    
     analyse_button = st.sidebar.button("Start Analysis")
     if analyse_button: 
          tweet_df = read_tweets_csv(dataset_file)
          st.write("The intial dataset")
          st.dataframe(tweet_df)
          # acquires both tweet polarity and subjectivity
          tweet_df['sentiment'] = tweet_df['text'].apply(lambda tweet: TextBlob(tweet).sentiment)
          # only polarity
          tweet_df['polarity_score'] = tweet_df['text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
          tweet_df['polarity'] = tweet_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
          
          
          
          tweet_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               colors=["limegreen", "red", "gray"])
          su=plt.axis('equal')
          su=plt.title("Franction of each sentiment in random tweets")
          su=plt.legend(labels=tweet_df.polarity.value_counts().index, loc="upper left")
          st.write(su.figure)
 
          tweets_dataset = tweet_df.copy()
          
          #Removing non-ascii characters (for example, arabian chars)
          tweet_df['text'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
          #Making all fields string type
          for i in range(len(tweet_df)): 
              tweet_df.at[i,'text'] = str(tweet_df.iloc[i]['text'])
          for i in range(len(tweet_df)): 
              tweet_df.at[i,'text'] = remove_urls(tweet_df.iloc[i]['text'])
          # Convert to list
          dataf= tweet_df['text'].values.tolist()
          dataf = [re.sub('\S*@\S*\s?', '', sent) for sent in dataf]
          dataf = [re.sub('\s+', ' ', sent) for sent in dataf]
          dataf = [re.sub("\'", "", sent) for sent in dataf]
          data_t = np.array(dataf)

          
          tokenizer = Tokenizer(num_words=max_words)
          test_sequence = tokenizer.texts_to_sequences(data_t)
          test_sequence = pad_sequences(test_sequence, maxlen=2500)
          test_prediction = model.predict(test_sequence)
          np.around(test_prediction, decimals=0)
          
          tweets_dataset['label'] = np.around(test_prediction, decimals=0)
          tweets_dataset[tweets_dataset['label']==1.0].head(10)
          
          for i in range(10):
              st.write(tweets_dataset.iloc[i*2]['text']) 
              st.write('\n')
       
