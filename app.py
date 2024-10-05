import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')



st.title("SMS Spam Classification")


# no need of this function because already data is preprocessing
# def transform_text(text):
    
#     text=text.lower()
#     text=nltk.word_tokenize(text)

#     y=[]
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text=y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text=y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))
            
    
#     return " ".join(y)
    


# load data and model
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

# input
input_sms=st.text_input("Enter the message")

if st.button("predict"):
    # perprocessing                # data is already preprocessing
    # transformed_text=input_sms

    # vectorize
    vector_input=tfidf.transform([input_sms])

    # predict
    result=model.predict(vector_input)[0]

    # display
    if result==1:
        st.header("spam")
    else:
        st.header("Not spam")
