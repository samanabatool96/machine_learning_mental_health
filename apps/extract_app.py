from All_files import *

def app():
    """## Tweets search Information """
    st.write("Fill labels with '*', if confused.") 
    words = st.text_input("Keywords/Hashtags/Usernames for twitter search *", "",)
    lang_query = ""
    since_query = ""
    until_query = ""
   
    if words:
        keyword_query = words
    lang = st.text_input("Select the language of the tweets to extract", value="en")
    if lang:
            lang_query = " lang:" + lang
    date_since = st.text_input("Extract tweets since (Format yyyy-mm-dd) * Recent search API allows only to get tweets of the previous 7 days", value="",)
    if date_since:
        since_query = " since:" + date_since
    date_untill = st.text_input("Extract tweets till (Format yyyy-mm-dd)", value="")
    if date_untill:
         until_query = " until:" + date_untill
               
         numtweet = st.text_input("Enter the number of tweets to be extracted (if not given default Max 15000) *",value="15000",)
         since_id = st.text_input("Extract tweets above this specific tweet id")
         filter = st.text_input("Enter any filter to be added for the search query")
         extract = st.button("Extract tweets")
         search_query = keyword_query + lang_query + since_query + until_query
         if extract:
                    tweets_csv_file = snscrape_func(search_query, int(numtweet))
                    b64 = base64.b64encode(
                        tweets_csv_file.encode()
                    ).decode()  # some strings <-> bytes conversions necessary here
                    """ ## Click the link below to download the Extracted tweets """
                    href = f'<a href="data:file/csv;base64,{b64}" download="extracted_tweets.csv">Download Extracted Tweets CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)