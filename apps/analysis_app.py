from All_files import *
def app():
    
    dataset_file = st.file_uploader(
                    "Upload Tweet Dataset", type=["csv"]
                )
   

    keyword_used = st.text_input(
                    "Enter the keyword used for extracting the tweet dataset",
                    key="keyword",
                )
    sentiment_analysis_list = ["Sentiment Wordclouds", "Sentiment Analysis"]            
    sentiment_analysis = st.multiselect(
                    "Select a list of sentiment analysis to compute",
                    sentiment_analysis_list,
                    default=sentiment_analysis_list,)           
          
    a_button = st.button("Start")
    if a_button:
        """# Read File """
        tweet_df = read_tweets_csv(dataset_file)
        
        st.title("Intial Dataset:")
        st.dataframe(tweet_df)
        st.title("Cleaned Dataset:")
        tweet_df["Mentioned_Hashtags"] = tweet_df["Text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
        tweet_df["Mentioned_Usernames"] = tweet_df["Text"].apply(
                        extract_username
                    )
        tweet_df["Clean Tweet"] = tweet_df["Text"].apply(clean_txt)
                   
                    

        st.write("The cleaned dataset")
        st.dataframe(tweet_df)
        st.write("Understanding dataset")
        """### Tweets wordcloud"""
        
        if (
                        "Sentiment Wordclouds" in sentiment_analysis
                        or "Sentiment Analysis" in sentiment_analysis
                    ):
                    if "sentiment" in tweet_df:
                        pass
                    else:
                        st.write("It might take several minutes to analyse the sentiments..." )
                        tweet_df = sentiments(tweet_df)
                        st.write("Sentiment Analysis Done on the tweets")
                        st.write(tweet_df)
                        b64 = base64.b64encode(tweet_df.to_csv().encode() ).decode()  # some strings <-> bytes conversions necessary here
                        """ ## Click the link below to download the Extracted tweets """
                        href = f'<a href="data:file/csv;base64,{b64}" download="extracted_tweets.csv">Download Tweets dataset with Sentiments CSV File for faster next time usage</a> (right-click and save as &lt;some_name&gt;.csv)'
                        st.markdown(href, unsafe_allow_html=True)

                       

                        """## Sentiment count plot"""
                        st.title("Sentiment Analysis:")
                        fig, ax = plt.subplots()
                        ax.tick_params(axis='x', colors='white')   
                        ax.tick_params(axis='y', colors='white')
                        ax.set_facecolor("#091C46")
                        fig.patch.set_facecolor('#091C46')
                        fig.set_size_inches(10, 8)
                        sns.countplot(
                            x=tweet_df["sentiment"], palette="Set3", linewidth=0.5
                        )
                        plt.xlabel('Sentiment',color="white")
                        plt.ylabel('count',color="white")
                        st.pyplot(fig)
                        plt.close()

                        # Create sentiment based tweets list

                        pos = []
                        neg = []
                        neu = []
                        for _, row in tweet_df.iterrows():
                            if row["sentiment"] == "positive":
                                pos.append(row["Clean Tweet"])
                            elif row["sentiment"] == "negative":
                                neg.append(row["Clean Tweet"])
                            elif row["sentiment"] == "neutral":
                                neu.append(row["Clean Tweet"])

                        """## Sentiment wordcloud"""
                        if "Sentiment Wordclouds" in sentiment_analysis:
                            st.title("Positive Wordcloud")
                            masked_worldcloud_generate(
                                list_data=pos,
                                file_path="icons/thumbs-up-solid.png",
                                background="#091C46",
                                color=color_dark28,
                                title="Positive sentiment word cloud on tweets",
                                font_path="font/BebasNeue-Regular.ttf",
                            )

                            st.title("Negative Wordcloud")

                            masked_worldcloud_generate(
                                list_data=neg,
                                file_path="icons/thumbs-down-solid.png",
                                background="#091C46",
                                color=grey_color_func,
                                title="Negative sentiment word cloud on tweets",
                                font_path="font/BebasNeue-Regular.ttf",
                            )

                            st.title("Neutral Wordcloud")

                            masked_worldcloud_generate(
                                list_data=neu,
                                file_path="icons/user-alt-solid.png",
                                background="#091C46",
                                color=grey_color_func,
                                title="Neutral sentiment word cloud on tweets",
                                font_path="font/BebasNeue-Regular.ttf",
                            )
                            """### Model Prediction"""
                            st.title("After Model Prediction")
                            data=tweet_df["Clean Tweet"]
                            tokenizer = Tokenizer(num_words=max_words)
                            tokenizer.fit_on_texts(data)
                            tweets = np.array(data)
                            sequences = tokenizer.texts_to_sequences(data)
                            tweets = pad_sequences(sequences, maxlen=max_len)
                        
                            predictions = model.predict(tweets)
                            

                            tweet_df['label'] = np.around(predictions, decimals=0)
                            st.write(tweet_df[tweet_df['label']==1.0].head(10))
                            
                            
                            fig = plt.figure(figsize=(20, 10))
                            ax = plt.axes()
                            ax.tick_params(axis='x', colors='white')   
                            ax.tick_params(axis='y', colors='white')
                            ax.set_facecolor("#091C46")
                            fig.patch.set_facecolor('#091C46')
                            tweet_df['label'].value_counts(normalize = True)
                            tweet_df['label'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
                            plt.xlabel('label',color="white")
                            plt.ylabel('count',color="white")
                            st.pyplot(fig)

                        

           