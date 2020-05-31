from time import time
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
import os


#load vectorizer
link_vectorizer = "final_tfidf"
vectorizer = joblib.load(link_vectorizer)

#load model
link_model = "model_final"
model = joblib.load(link_model)

#load data
test_path = '/vietnamese_news_ml_dl/10topics_update_final/test10_origin_processed/'
test_data = []
label_test  = []

#load encoder
le_path = "labelEncode"
le = joblib.load(le_path)

for topic in os.listdir(test_path):

    pd = os.path.join(test_path, topic)
    for file in os.listdir(pd):
        pf = os.path.join(pd, file)
        text = open(pf, 'r',encoding="utf-8").read()

        test_data.append(text)
        label_test.append(topic)

label_test = le.transform(label_test)
test_data_transform = vectorizer.transform(test_data)

#predict
time_start = time()
label_predict = model.predict(test_data_transform)
time_end = time()

name = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe','The gioi', 'The thao', 'Van hoa', 'Vi tinh']
print(classification_report(label_test,label_predict,target_names=name))
print(confusion_matrix(label_test,label_predict))
print(f"predict time: {time_end-time_start} s")