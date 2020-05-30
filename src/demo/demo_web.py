import re
import joblib
import numpy as np
import streamlit as st
from urllib import request
from utils import visualize
# from topic_news_classification.prepro_code.preprocess import Prepro # streamlit bug here, add path may fix
from preprocess import Prepro
# import pandas as pd
# from gensim.utils import simple_preprocess
# import matplotlib.pyplot as plt

PATH_STOPSWORD = r"../../data/vietnamese_stopsword"
PATH_ACRONYM = r"../../data/vietnamese_tvt"

PATH_SVM = r"../../models/svm_classifier"
PATH_NB = r"../../models/nb_classifier"
PATH_LR = r"../../models/lr_classifier"
PATH_TFIDF = r"../../models/tfidf_vectorizer"
PATH_ENCODER = r"../../models/label_encoder"
# create/load model


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    """"load all pretrain/object needed"""
    prepro = Prepro(PATH_STOPSWORD, PATH_ACRONYM)
    vectorizer = joblib.load(PATH_TFIDF)
    label_encoder = joblib.load(PATH_ENCODER)
    model_svm = joblib.load(PATH_SVM)
    model_nb = joblib.load(PATH_NB)
    model_lr = joblib.load(PATH_LR)
    return prepro, vectorizer, label_encoder, model_svm, model_nb, model_lr


prepro, vectorizer, label_encoder, model_svm, model_nb, model_lr = load_model()

ALL_LABELS = label_encoder.classes_
st.title('Demo News classification')

options = ["Links", "Paragraph"]
choice = st.selectbox("Link or paragraph?", options)
raw_data = ""
if choice == options[0]:
    link = st.text_input("", 'Enter your link here!')
    try:
        with request.urlopen(link) as response:
            html = response.read().decode('utf-8')

        data = re.findall("<p>.*</p>", html)
        # data_show = [" ".join(simple_preprocess(txt)) for txt in data]
        data = [d.replace("<p>", " ").replace("</p>", ".") for d in data]
        raw_data += " ".join(data)
        # st.write(data_show)
        st.text_area("", raw_data)
    except:
        pass
else:
    raw_data = st.text_area("Enter your paragraph!")

clear_data = prepro.normalize(raw_data)
vector_data = vectorizer.transform([clear_data]).toarray()
# predict prob
# [0] below for test only one data
svm = model_svm.predict_proba(vector_data)[0]
logistic = model_lr.predict_proba(vector_data)[0]
nb = model_nb.predict_proba(vector_data)[0]

# average prob
total_prob = (svm + logistic + nb) / 3
# get max prob
idx = np.argmax(total_prob)
label_pred = label_encoder.inverse_transform([idx])[0]
prob_pred = total_prob[idx]
# classification with raw data here
# return two array
# namegroups: le.classes_, name of all class
# return 3 array of predict prob
st.write("Result: **{}** with confident: **{:.2f}**".format(label_pred, prob_pred))
st.write("\n\n\n")
something = ["OK", "More detail", "Relabel"]
result_satisfy = st.selectbox("", something)

if result_satisfy == something[2]:
    new_label = st.selectbox("New label", ALL_LABELS)
    st.write("*Label this paragraph as __{}__*".format(new_label))
elif result_satisfy == something[1]:
    labels = ["svm", "logistic", "naive bayes"]
    name_groups = ALL_LABELS  # input, gettop=> name_change here, data.
    groups = [svm, logistic, nb]
    visualize(groups, name_groups, labels, top_k=10, width=.35, gap=.3)

    st.pyplot()

if __name__ == "__main__":
    print("Still running......")