from __future__ import unicode_literals
import pandas as pd
import re
import os
import io
import numpy as np
import seaborn as sns
import spacy
import en_core_web_sm
import snscrape.modules.twitter as sntwitter



import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# from tqdm.notebook import tqdm
import nltk
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from wordcloud import WordCloud
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from streamlit_disqus import st_disqus
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import random
import ast
from collections import Counter
from palettable.colorbrewer.qualitative import Dark2_8
from palettable import cubehelix
from matplotlib import cm
from collections import Counter
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# from transformers import *
import streamlit as st
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
import textblob_de
from textblob_de import TextBlobDE

import scattertext as sctxt
import streamlit.components.v1 as components

import hashlib
from streamlit_disqus import st_disqus
import keras
from keras.models import load_model
import ftfy
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
max_words = 20000
max_len = 400
tokenizer = Tokenizer(num_words=max_words)


#import tensorflow as tf

st.set_option("deprecation.showfileUploaderEncoding", False)

nltk.download("words")
nltk.download("punkt")
nltk.download("stopwords")

english_stop_word = stopwords.words("english")
stopwords = english_stop_word
stopwords.append("nan")
model =keras.models.load_model('complete_model/')
matplotlib.rc("font", **{"sans-serif": "Arial", "family": "sans-serif"})
def load_file(filename):
    df = pd.read_csv(filename, index_col=0)
    return df
nlp = en_core_web_sm.load()
class MultiApp:
    
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # app = st.sidebar.radio(
        app = st.selectbox(
            'Analytics Choice',
            self.apps,
            format_func=lambda app: app['title'])
        app['function']()
        

# sentiment = pipeline("sentiment-analysis")  # For english
analyzer = SentimentIntensityAnalyzer()
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def read_tweets_csv(file_path):
    df = pd.read_csv(file_path,encoding='latin1')
    return df


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def extract_hashtag(input_text):
    hashtags = re.findall(r"#[\w]*", str(input_text))
    return hashtags


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def extract_username(input_text):
    usernames = re.findall(r"@[\w]*", str(input_text))
    return usernames


def remove_urls(s):
    # re.sub(pattern,repl,string) is used to replace substrings. Will replace the matches in string with repl
    return re.sub(r"https?://\S+", "", s)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def clean_txt(input_text):
    input_text = str(input_text)
    if type(input_text) != str:
        return input_text
    # removing hashtags,emojis,stopwords
    input_text = re.sub(r"#[\w]*", "", str(input_text))
    input_text = input_text.encode("ascii", "ignore")
    input_text = input_text.decode()
    input_text = remove_urls(input_text)

    ##removing @user
    r = re.findall(r"@[\w]*", input_text)
    for i in r:
        input_text = re.sub(i, "", input_text)
        input_text = re.sub("amp", "", input_text)
        input_text = re.sub("nan", "", input_text)

    # removing stopwords
    input_text = " ".join(
        [i for i in wordpunct_tokenize(input_text.lower()) if not i in stopwords]
    )

    # removing punctuation,numbers and whitespace
    result = re.sub(r"[^\w\s]", "", input_text.lower())
    result = re.sub("\s+", " ", result)
    ##removing links
    rresultes = re.sub(r"https[\w]*", "", result, flags=re.MULTILINE)
    # removing digits
    result = "".join(i for i in result if not i.isdigit())

    return result        
def plot_countplots(y_value, data_df, x_counts, title):
    sns.set()
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    g = sns.countplot(y=y_value, data=data_df, order=x_counts)
    plt.title(title)
    plt.close()
    st.pyplot(fig)


def grey_color_func(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


def color_dark28(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Dark2_8.colors[random.randint(0, 7)])


def color_cubehelix(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return tuple(cubehelix.perceptual_rainbow_16_r.colors[random.randint(0, 7)])


def masked_worldcloud_generate(
    list_data, file_path, font_path, background, title, color="None"
):
    wcloud = " ".join([i for i in list_data if type(i) == str])
    icon = Image.open(file_path)
    mask = Image.new("RGB", icon.size, (255, 255, 255))
    mask.paste(icon, icon)
    mask = np.array(mask)
    wordcloud = WordCloud(
        random_state=42,
        max_font_size=500,
        font_path=font_path,
        mask=mask,
        background_color=background,
        stopwords=stopwords,
        repeat=False,
    ).generate(wcloud)
    image_colors = ImageColorGenerator(mask)
    fig = plt.figure(figsize=[15, 10])
    if color != None:
        plt.imshow(wordcloud.recolor(color_func=color, random_state=3))
    elif color == "None":
        plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)
    plt.close()


def worldcloud_generate(list_data, background, title, font_path, color):
    wcloud = " ".join([i for i in list_data if type(i) == str])
    wordcloud = WordCloud(
        random_state=42,
        max_font_size=300,
        background_color=background,
        font_path=font_path,
        color_func=color,
        repeat=False,
        stopwords=stopwords,
        relative_scaling="auto",
    ).generate(wcloud)
    fig = plt.figure(figsize=[12, 8])
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)
    plt.close()




@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def snscrape_func(search_query, num_tweet):
    st.write(search_query)
    columns = [
        "Tweet Id",
        "DateTime",
        "Username",
        "Text"
       
        
    ]
    tweets_list = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
        if i > num_tweet:
            break
        tweets_list.append(
            [
                tweet.id,
                tweet.date,
                tweet.username,
                tweet.content
               
                

            ]
        )

    # Creating a dataframe from the tweets list above
    tweets_df = pd.DataFrame(tweets_list, columns=columns)
    st.dataframe(tweets_df)
    tweets_csv_file = tweets_df.to_csv(index=True)
    return tweets_csv_file


# def get_scores(text):
#     txt = sentiment(text)
#     return txt[0]["label"]


def convert_lower(text):
    res = text.lower()
    return res




from collections import Counter


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def polarity_plot(df, title):
    length = df.shape[0]
    if length < 500:
        fig, ax = plt.subplots()
        g = sns.barplot(x=df.index.values, y="Polarity", data=df, palette="rocket")
        plt.title(title)
        st.pyplot(fig)
        plt.close()
    else:
        st.write("If it is a large dataset, only a sample is shown in the graph")
        df = df.sample(500, random_state=2)
        fig, ax = plt.subplots()
        g = sns.barplot(x=df.index.values, y="Polarity", data=df, palette="rocket")
        plt.title(title)
        st.pyplot(fig)
        plt.close()


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def tweets_on_dates(df, title):

    # st.write("If it is a large dataset, only a sample is shown in the graph")

    date_list = []
    count_list = []
    length = df.shape[0]
    dates = df["date"].dt.date
    tweet_count = Counter(list(dates))
    for i, v in tweet_count.items():
        date_list.append(i)
        count_list.append(v)
    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(12, 12))

    # Add x-axis and y-axis
    # value = int(len(date_list) / 4)
    # ax.bar(date_list[:value], count_list[:value])
    ax.bar(date_list, count_list)

    # Set title and labels for axes
    ax.set(xlabel="Date", ylabel="Counts", title=title)

    plt.xticks(rotation=90)
    # Define the date format
    date_form = DateFormatter("%d-%m-%y")
    ax.xaxis.set_major_formatter(date_form)

    # Ensure a major tick for each week using (interval=1)
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    st.pyplot(fig)
    plt.close()


def sentiments_on_dates(df, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    g = sns.countplot(
        df["date"].dt.date, hue=df["sentiment"], palette="Set3", linewidth=0.5
    )
    plt.xticks(rotation=90)
    plt.title(title)
    st.pyplot(fig)
    plt.close()


def list_hashtags_usernames(df):
    hashtags = df["Mentioned_Hashtags"]
    usernames = df["Mentioned_Usernames"]

    HT_list = sum(list(hashtags), [])
    UN_list = sum(list(usernames), [])

    return HT_list, UN_list


def sentiment_hashtags_usernames(df):
    HT_positive = []
    HT_negative = []
    HT_neutral = []
    UN_positive = []
    UN_negative = []
    UN_neutral = []
    positive_retweets = []
    negative_retweets = []
    neutral_retweets = []

    sentiment_df = df.groupby("sentiment")

    keys = sentiment_df.groups.items()

    for i, v in keys:

        if i == "positive":

            positive_df = sentiment_df.get_group("positive")
            hashtags_positive = positive_df["Mentioned_Hashtags"]
            HT_positive = sum(list(hashtags_positive), [])
            usernames_positive = positive_df["Mentioned_Usernames"]
            UN_positive = sum(list(usernames_positive), [])
            positive_retweets = positive_df["retweetcount"]

        elif i == "negative":

            negative_df = sentiment_df.get_group("negative")
            hashtags_negative = negative_df["Mentioned_Hashtags"]
            HT_negative = sum(list(hashtags_negative), [])
            usernames_negative = negative_df["Mentioned_Usernames"]
            UN_negative = sum(list(usernames_negative), [])
            negative_retweets = negative_df["retweetcount"]

        elif i == "neutral":

            neutral_df = sentiment_df.get_group("neutral")
            hashtags_neutral = neutral_df["Mentioned_Hashtags"]
            HT_neutral = sum(list(hashtags_neutral), [])
            usernames_neutral = neutral_df["Mentioned_Usernames"]
            UN_neutral = sum(list(usernames_neutral), [])
            neutral_retweets = neutral_df["retweetcount"]

    return (
        HT_positive,
        HT_negative,
        HT_neutral,
        UN_positive,
        UN_negative,
        UN_neutral,
        positive_retweets,
        negative_retweets,
        neutral_retweets,
    )


def plot_hash_user_count(all, positive, neutral, negative, common, title):
    color = cm.winter_r(np.linspace(0.5, 0.2, 20))
    hashtag_count_dict = {
        "Total ": len(all),
        "Positive ": len(positive),
        "Neutral ": len(neutral),
        "Negative ": len(negative),
        "Common ": len(common),
    }
    fig = plt.figure(figsize=(10, 8))
    lists = sorted(
        hashtag_count_dict.items(), reverse=True
    )  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    # for i, v in enumerate(y):
    #     plt.text(i - 0.15, v / y[i] + 600, y[i], color="white", fontsize=12)

    plt.bar(x, y, color=color, width=0.5)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontdict={"fontsize": 22})
    plt.ylabel("Counts", fontdict={"fontsize": 22})
    plt.xlabel("Values", fontdict={"fontsize": 22})
    st.pyplot(fig)
    plt.close()


def plot_freq_dist(list_data, title, n):
    a = nltk.FreqDist(list_data)
    d = pd.DataFrame({"Key": list(a.keys()), "Value": list(a.values())})
    # selecting top 10 most frequent hashtags
    d = d.nlargest(columns="Value", n=n)
    fig = plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=d, x="Key", y="Value")
    ax.set(ylabel="Count")
    plt.xticks(rotation=90)
    plt.title(title)
    st.pyplot(fig)
    plt.close()


def plot_retweet_count(x_value, title):
    fig = plt.figure(figsize=(16, 5))
    sns.countplot(x=x_value)
    plt.xlim(right=9)
    plt.title(title)
    st.pyplot(fig)
    plt.close()


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def sentiment_scores(text):
    analysis = TextBlob(text, analyzer=NaiveBayesAnalyzer()).sentiment

    sentiment = analysis[0]
    if sentiment == "pos":
        sentiment = "positive"
    else:
        sentiment = "negative"

    return sentiment


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def get_polarity_de(text):
    return TextBlobDE(text).sentiment.polarity


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def sentiments(df):

    st.write("Using Text Blob")
    df["Polarity"] = df["Clean Tweet"].apply(get_polarity)
    df["sentiment"] = ""
    df.loc[df.Polarity > 0, "sentiment"] = "positive"
    df.loc[df.Polarity == 0, "sentiment"] = "neutral"
    df.loc[df.Polarity < 0, "sentiment"] = "negative"
    # sentiments_progress = st.progress(0)
    # sentiments_list = []
    # for i, row in df.iterrows():
    #     sentiments_list.append(sentiment_scores(row["Clean Tweet"]))
    #     sentiments_progress.progress(i / df.shape[0])
    # df["sentiment"] = sentiments_list
    return df



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def scatterplot_sentiment_log_scale_phrase_plot(df):

    corpus = sctxt.CorpusFromPandas(
        df,
        category_col="sentiment",
        text_col="Clean Tweet",
        nlp=sctxt.whitespace_nlp_with_sentences,
    ).build()

    html = sctxt.produce_scattertext_explorer(
        corpus,
        category="positive",
        category_name="Positive",
        not_category_name="Negative",
        neutral_category_name="Neutral",
        minimum_term_frequency=5,
        width_in_pixels=1000,
        transform=sctxt.Scalers.log_scale_standardize,
    )

    return html


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def scatterplot_sentiment_bm25_visualisation(df):

    corpus = (
        sctxt.CorpusFromPandas(
            df,
            category_col="sentiment",
            text_col="Clean Tweet",
            nlp=sctxt.whitespace_nlp_with_sentences,
        )
        .build()
        .get_unigram_corpus()
    )

    term_scorer = sctxt.BM25Difference(corpus, k1=1.2, b=0.9).set_categories(
        "positive", ["negative"], ["neutral"]
    )

    html = sctxt.produce_frequency_explorer(
        corpus,
        category="positive",
        not_categories=["negative"],
        neutral_categories=["neutral"],
        term_scorer=term_scorer,
        # metadata=df["username"],
        grey_threshold=0,
        show_neutral=False,
    )

    return html


def sentiment_topic_analysis(df):

    # chose the category to compare
    pair = "positive", "negative"
    df_pair = df[df["sentiment"].isin(pair)]
    df_pair["Clean Tweet"] = df_pair["Clean Tweet"].apply(nlp)

    # create a corpus of extracted topics
    feat_builder = sctxt.FeatsFromOnlyEmpath()
    empath_corpus_pair = sctxt.CorpusFromParsedDocuments(
        df_pair,
        category_col="sentiment",
        feats_from_spacy_doc=feat_builder,
        parsed_col="Clean Tweet",
    ).build()

    # visualize Empath topics
    html = sctxt.produce_scattertext_explorer(
        empath_corpus_pair,
        category="positive",
        category_name="Positve",
        not_category_name="Negative",
        neutral_category_name="Neutral",
        width_in_pixels=1000,
        use_non_text_features=True,
        use_full_doc=True,
        topic_model_term_lists=feat_builder.get_top_model_term_lists(),
    )

    return html

