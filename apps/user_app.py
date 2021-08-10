from matplotlib import colors, use
from All_files import *


def app():
     user = ["Alee", "Ayesha", "Gillian Jones", "Jack","Joe Hill","Kai","Stephen King","Melanin","Mr Cello","Rohit Sharma","Samantha"]
     user_choice = st.selectbox("Username", user)
     if user_choice== "Alee":
                user_df = load_file("./datasets/Alee.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")

                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)

                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)
                    
                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    """#This will take a few minutes"""
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                    """#This will take a few minutes"""   
                    predictions = model.predict(tweets)
                  

                    user_df['label'] = np.around(predictions, decimals=0)
                    st.write(user_df[user_df['label']==1.0].head(10))
                    
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                     
                   
     elif user_choice == "Ayesha":
                user_df = load_file("./datasets/Ayesha.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)

                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)      
                    predictions = model.predict(tweets)
              
                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                    
     if user_choice == "Gillian Jones":
                user_df = load_file("./datasets/Gillian_James.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)

                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                        
                    predictions = model.predict(tweets)
                    

                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                       
     if user_choice == "Jack":
                user_df = load_file("./datasets/jack.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)

                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                        
                    predictions = model.predict(tweets)
                        
                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                       
     if user_choice == "Joe Hill":
                user_df = load_file("./datasets/Joe Hill.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)

                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                        
                    predictions = model.predict(tweets)
                    

                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                       
     if user_choice == "Kai":
                user_df = load_file("./datasets/kai.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)
                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)


                    predictions = model.predict(tweets)
                

                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                     
     if user_choice == "Stephen King":
                user_df = load_file("./datasets/king.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)
                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                        
                    predictions = model.predict(tweets)
                    

                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                                                                                                                                                      
     if user_choice == "Mr Cello":
                user_df = load_file("./datasets/mr_Cello.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    
                    st.write("Cleaned Datasets!!")

                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    

                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30)
                    
                    st.pyplot(plt)


                    data=user_df["Clean Tweet"]
                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)
                    """#This will take a few minutes"""
                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                        
                    predictions = model.predict(tweets)
                   

                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
                    
     if user_choice == "Rohit Sharma":
        
                user_df = load_file("./datasets/rohit_sharma.csv")
                st.write("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)
                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")
                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                        
                    predictions = model.predict(tweets)
                    
                    st.write("10 Depressive Tweets")
                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
              
                                          
     if user_choice == "Samantha":
                user_df = load_file("./datasets/samantha.csv")
                st.title("The intial dataset")
                st.dataframe(user_df)
                clean_button = st.button("Start Analysing")
                if clean_button:
                    """# Data Cleaning"""
                    user_df["Mentioned_Hashtags"] = user_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    user_df["Mentioned_Usernames"] = user_df["Text"].apply(
                        extract_username
                    )
                    user_df["Clean Tweet"] = user_df["Text"].apply(clean_txt)
                    data=user_df["Clean Tweet"]
                    st.write("Cleaned Datasets!!")
                    
                    """#This will take a few minutes"""
                    user_df['sentiment'] = user_df["Clean Tweet"] .apply(lambda tweet: TextBlob(tweet).sentiment)
                    # only polarity
                    user_df['polarity_score'] = user_df["Clean Tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    user_df['polarity'] = user_df['polarity_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                    
                    st.title("Sentiment Polarity Chart")
                    fig, ax = plt.subplots(figsize = (20,12))
                    user_df.polarity.value_counts().plot(kind="pie",
                               autopct='%1.1f%%',
                               labels=None,
                               pctdistance=1.12,
                               textprops={'color':"w",'fontsize': 30},
                               colors=["limegreen", "red", "gray"])
                    fig.set_facecolor("#091C46")  
                    ax=plt.axis('equal')
                    plt.legend(labels=user_df.polarity.value_counts().index, loc="upper left",fontsize =30) 
                    st.pyplot(plt)
                    user_df['subjectivity_score'] =  user_df['Text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                    user_df['subjectivity'] = user_df['subjectivity_score'].apply(lambda x: 'subjective' if x>0.5 else ('objective' if x < 0.5 else 'neutral'))
                    likesorder=user_df[['Datetime','Likes_Count','polarity_score','Text']].sort_values(by=['Likes_Count'],ascending=False)
                    
                    
                    
                    user_df['datetime'] = pd.to_datetime(user_df['Datetime'])
                    user_df = user_df.set_index('datetime')
                    

                    hours_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.hour).mean()
                   
                    st.title('Average Tweet Likes by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['Likes_Count'],color='blue',marker='o')
                    plt.xlabel('Hour of the Day',color="white")
                    plt.ylabel('average likes count',color="white") 
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Hour of the Day')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(hours_avg['polarity_score'],color='green',marker='o')
                    plt.xlabel('Hour of the Day')
                    st.pyplot(plt) 
            
                    months_count=user_df.resample('M').count()
              
                    st.title('Tweets by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_count['Username'],color='blue',marker='o')
                    st.pyplot(plt)
                    
                    months_avg=user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].resample('M').mean()
                    
                    st.title('Average Tweet Likes by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)


                    st.title('Average Tweet Sentiment Score by Month')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(months_avg['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)

                    week_df = user_df[['retweet_Count','Likes_Count','polarity_score','subjectivity_score']].groupby(user_df.index.day_name()).mean()
                    # label each day of week in order so it will be sorted this way rather than alphabetically
                    week_df.at['Sunday', 'sort'] = 1
                    week_df.at['Monday', 'sort'] = 2
                    week_df.at['Tuesday', 'sort'] = 3
                    week_df.at['Wednesday', 'sort'] = 4
                    week_df.at['Thursday', 'sort'] = 5
                    week_df.at['Friday', 'sort'] = 6
                    week_df.at['Saturday', 'sort'] = 7
                    week_df=week_df.sort_values(by=['sort'])

                    st.title('Average Tweet Likes by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['Likes_Count'],color='blue',marker='o')
                    st.pyplot(plt)

                    st.title('Average Tweet Sentiment Score by Day of the Week')
                    fig=plt.figure(figsize=(12,9))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    plt.plot(week_df['polarity_score'],color='green',marker='o')
                    st.pyplot(plt)
                   
                    
                    st.title("Prediction on the tweets")
                    st.write("where 1 represents Depressed and Anxious Tweet and 0 represents Non-Depressive Tweets")

                    tokenizer = Tokenizer(num_words=max_words)
                    tokenizer.fit_on_texts(data)
                    tweets = np.array(data)
                    sequences = tokenizer.texts_to_sequences(data)
                    tweets = pad_sequences(sequences, maxlen=max_len)
                        
                    predictions = model.predict(tweets)
                    st.write("10 Depressive Tweets")

                    user_df['label'] = np.around(predictions, decimals=0)
                    user_df[user_df['label']==1.0].head(10)
              
                    fig = plt.figure(figsize=(20, 10))
                    ax = plt.axes()
                    ax.tick_params(axis='x', colors='white')   
                    ax.tick_params(axis='y', colors='white')
                    ax.set_facecolor("#091C46")
                    fig.patch.set_facecolor('#091C46')
                    user_df['label'].value_counts(normalize = True)
                    user_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                    plt.xlabel('label',color="white")
                    plt.ylabel('count',color="white")
                    st.pyplot(fig)
