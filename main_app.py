from hashlib import algorithms_guaranteed
from seaborn.palettes import color_palette
from All_files import *
from All_files import MultiApp
from apps import analysis_app,extract_app,user_app

app=MultiApp()
with open("styles/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
side_bar=st.beta_container()
col2 = st.beta_container()

col2.image('images/0001.jpg')
   
  
with side_bar:
    st.sidebar.image("images/circle-cropped.png", width=250)

menu = ["Home","How it Works","Learn","Main Analysis"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.markdown('#') 
  
    st.markdown("<h1 style='text-align:center;font-weight:bolder;font-size:60px; color: #356E98;  font-family: Freestyle Script'>Every life is believed to be like a light and every life that goes out much too soon is previous. We lost many great people due to suicide and this alarming and needs to be changed.It is mandatory to find out people who are suffering and this is considered to be the first phase that is this whole project is based on.</h1>", unsafe_allow_html=True)
    
    st.markdown('#') 
    st.markdown('#') 
    st.markdown('#') 
    st.markdown("<h1 style='text-align:left;font-weight:bolder;font-size:20px; color: white;  font-family: 	Copperplate'> <<<< Click the arrow on the top left to open sidebar</h1>", unsafe_allow_html=True)

elif choice == "How it Works":
    """
    ## Steps
    1. **Extract Tweets**: 
       Step1: write keywords which includes Username, Hashtag, Word,Emoji etc
       #
       Step2: Select Language. By default it will say 'en' because the whole model is trained on english language.
       Step3: Enter date.
       Step4: Check on to sentiment and then press the button. It will take few minutes.
       Step5: Dataset is prepare you can download for further analysis.

    2. **Analyse Tweets** - creating a tweet analysis report for the saved datasets.
       Step1: Upload Dataset of your own choice.
       Step2: Click on "Start Analysing" Button.
       Step3: Datasets is being cleaned will take time depending on the size of datasets.
       Step4: Sentiments are added using TextBlob.
       Step5: Trained Model Predicts the depression on the Dataset Provided

    3. **User Analysis** - creating a tweet analysis report for the saved datasets.
       Step1: Choose Username 
       #
       Step2: Click on "Start Analysing" Button.
       Step3: Datasets is being cleaned will take time depending on the size of datasets.
       Step4: Sentiments are added using TextBlob.
       Step5: Trained Model Predicts the depression on the Dataset Provided
    
    """ 

elif choice == "Learn More":
    """
    st.markdown("<h1 style='text-align:center;font-weight:bolder;font-size:50px; color: white;  font-family: 	Copperplate'>Learn More</h1>", unsafe_allow_html=True)
    ### DATA that will talk to you if you're willing to Listen...
    """
    st.write(
        """
           Web-scrapping and Data Analytics falls under my project domain. Web scraping, web mining, or web harvesting are data mining techniques that collect or extract large amounts of data from various websites or social media. It is usually accomplished using scripts that interface with web through browser or direct protocols like html etc. The ‘scrapped’ or extracted data is pre-processed and archived as structured or unstructured database. We intend using Twitter as data source and Twitter APIs for extracting the data.  
           Data Analytics is a wild field encompassing various techniques. Under the project we intend using Natural Language Processing (NLP) and Machine Learning (ML) to classify the tweets as per the mood and emotion.  
        """
    )
    st.write("#")

    st.write("It was estimated by the WHO and further analised that ")
    data=load_file("./datasets/who_suicide_statistics.csv")
    st.title('Distribution of suicides from the year 1985 to 2016')
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()
    ax.tick_params(axis='x', colors='white')   
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor("#091C46")
    fig.patch.set_facecolor('#091C46')
    data['year'].value_counts(normalize = True )
    data['year'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (8, 6))


    plt.xlabel('year',color="white")
    plt.ylabel('count',color="white")
    st.pyplot(fig)
elif choice =="Main Analysis":
    st.markdown("<h1 style='text-align:center;font-weight:bolder;font-size:50px; color: white;  font-family: 	Copperplate'>Twitter Analytics Option</h1>", unsafe_allow_html=True)
    app.add_app("Extract Tweets",extract_app.app)
    app.add_app("Analysis Datasets",analysis_app.app)
    app.add_app("User Analysis",user_app.app)
    
    app.run()



     