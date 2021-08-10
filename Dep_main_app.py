from All_files import *
from All_files import AllLib
side_bar=st.beta_container()

with side_bar:
    st.sidebar.image("images/Logo_v.png", width=250,)

def load_data():
    df = pd.read_csv('combined_csv.csv')
    df = df.iloc[:,1:]
    return df

#loading the data
df = load_data()

#intializing the GotLib object
got = AllLib(df) 

class select_box:
    value="StephenKing"
    def __init__(self,data):
        self.data=data
        self.box=None
    def place(self,title,key):
        header(title)
        self.box = st.selectbox(str(key),self.data)
        select_box.value=self.box    
def title(text,size,color):
    st.markdown(f'<h1 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h1>',unsafe_allow_html=True)

def header(text):
    st.markdown(f"<p style='color:white;'>{text}</p>",unsafe_allow_html=True)

with open("styles/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)    

title("Username Analysis",60,"white")


users = got.get_data_tweets()
stb1 = select_box(users)
stb1.place("Username",0)
@st.cache(persist=True)
def sbyc(df,stb1):
    return got.show_bar_by_user_tweet(stb1)

t_data = sbyc(df,stb1.value)
data_tweets_arr = [x for x in stb1['Text']]
X_d = clean_tweets(data_tweets_arr)
st.write(X_d)