import re
import joblib
import numpy as np
from urllib import request
# from topic_news_classification.prepro_code.preprocess import Prepro # streamlit bug here, add path may fix
from preprocess import Prepro

PATH_STOPSWORD = r"../../data/vietnamese_stopsword"
PATH_ACRONYM = r"../../data/vietnamese_tvt"

PATH_SVM = r"../../models/svm_classifier"
PATH_NB = r"../../models/nb_classifier"
PATH_LR = r"../../models/lr_classifier"
PATH_TFIDF = r"../../models/tfidf_vectorizer"
PATH_ENCODER = r"../../models/label_encoder"

if __name__ == "__main__":
    prepro = Prepro(PATH_STOPSWORD, PATH_ACRONYM)
    vectorizer = joblib.load(PATH_TFIDF)
    label_encoder = joblib.load(PATH_ENCODER)
    model_svm = joblib.load(PATH_SVM)
    model_nb = joblib.load(PATH_NB)
    model_lr = joblib.load(PATH_LR)

    while True:
        # input data
        p = input("Enter your paragraph: ")
        if p == "":
            break
        clear_data = prepro.normalize(p)
        vector_data = vectorizer.transform([clear_data]).toarray()
        # predict
        svm = model_svm.predict_proba(vector_data)[0]
        logistic = model_lr.predict_proba(vector_data)[0]
        nb = model_nb.predict_proba(vector_data)[0]
        # get result svm
        idx = np.argmax(svm)
        label_svm = label_encoder.inverse_transform([idx])[0]
        prob_svm = svm[idx]
        # get result logistic regression
        idx = np.argmax(logistic)
        label_lr = label_encoder.inverse_transform([idx])[0]
        prob_lr = logistic[idx]
        # get result naive bayes
        idx = np.argmax(nb)
        label_nb = label_encoder.inverse_transform([idx])[0]
        prob_nb = nb[idx]

        # print result
        print("svm predict topic of this paragraph: {:^20s} with {:.2f} confident".format(label_svm, prob_svm))
        print("naive bayes predict topic of this paragraph: {:^20s} with {:.2f} confident".format(label_nb, prob_nb))
        print("logistic regression predict topic of this paragraph: {:^20s} with {:.2f} confident".format(label_lr, prob_lr))
        print("\n\n")