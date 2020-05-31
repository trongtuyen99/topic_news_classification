# clone data
#!git clone https://github.com/trongtuyen99/vietnamese_news_ml_dl


# preprocess data for train and test
import os
#%cd /content/vietnamese_news_ml_dl
# training data: 
path = r'./10topics_update_final/train_0.8'
X_train10 = []
Y_train10 = []
# for train
for topic in os.listdir(path):
  pd = os.path.join(path, topic)
  for file in os.listdir(pd):
    pf = os.path.join(pd, file)
    text = open(pf, 'r', encoding="utf8").read()
    X_train10.append(text)
    Y_train10.append(topic)


# validation data:
path = r'./10topics_update_final/valid_0.8'
X_valid10 = []
Y_valid10 = []
# for valid
for topic in os.listdir(path):
  pd = os.path.join(path, topic)
  for file in os.listdir(pd):
    pf = os.path.join(pd, file)
    text = open(pf, 'r', encoding="utf8").read()
    X_valid10.append(text)
    Y_valid10.append(topic)


# test data: 
path = r'./10topics_update_final/test10_origin_processed'
X_test10 = []
Y_test10 = []
# for test
for topic in os.listdir(path):
  pd = os.path.join(path, topic)
  for file in os.listdir(pd):
    pf = os.path.join(pd, file)
    text = open(pf, 'r', encoding="utf8").read()
    X_test10.append(text)
    Y_test10.append(topic)


# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import preprocessing

count_vector = CountVectorizer()
X_tranform10 = count_vector.fit(X_train10)
X_tranform10 = count_vector.transform(X_train10)
# label encoder
le = preprocessing.LabelEncoder()
le.fit(Y_train10)
Y_transform10 = le.transform(Y_train10)

# for validation data
X_valid_transform10 = count_vector.transform(X_valid10)
Y_valid_transform10 = le.transform(Y_valid10)

# for test data
X_test_transform10 = count_vector.transform(X_test10)
Y_test_transform10 = le.transform(Y_test10)


# Tf-idf Vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import preprocessing

# for train data
tf_transformer = TfidfTransformer(use_idf=True).fit(X_tranform10)
X_train10_tf = tf_transformer.transform(X_tranform10)

# for valid data
X_valid10_tf = tf_transformer.transform(X_valid_transform10)

# for test data
X_test10_tf = tf_transformer.transform(X_test_transform10)


# Truncated SVD for using tf-idf
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=1000)
svd.fit(X_train10_tf)

X_train_transform10_pca = svd.transform(X_train10_tf)
X_valid_transform10_pca = svd.transform(X_valid10_tf)
X_test_transform10_pca = svd.transform(X_test10_tf)


# Best model using tf-idf
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

model_svm = SVC(kernel='linear', gamma='auto', C=6.5)
model_svm.fit(X_train_transform10_pca, Y_transform10)
y_train_predict10 = model_svm.predict(X_train_transform10_pca)
y_valid_predict10 = model_svm.predict(X_valid_transform10_pca)
y_test_predict10 = model_svm.predict(X_test_transform10_pca)
print('kernel = linear, gamma = auto, C = 6.5')
print(accuracy_score(y_train_predict10, Y_transform10))
print(accuracy_score(y_valid_predict10, Y_valid_transform10))
print(accuracy_score(y_test_predict10, Y_test_transform10))
print(classification_report(Y_test_transform10, y_test_predict10))
print(confusion_matrix(Y_test_transform10, y_test_predict10))