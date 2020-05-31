import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from time import time


"""read data"""
link_data_train = "vietnamese_news_ml_dl/10topics_update_final/train_0.8/"
link_data_valid = "vietnamese_news_ml_dl/10topics_update_final/valid_0.8/"

path = link_data_train
X_train = []
Y_train = []
for topic in os.listdir(path):
    pd = os.path.join(path, topic)
    for file in os.listdir(pd):
        print(file)
        pf = os.path.join(pd, file)
        text = open(pf, 'r',encoding="utf-8").read()

        X_train.append(text)
        Y_train.append(topic)

path = link_data_valid
X_valid = []
Y_valid  = []

for topic in os.listdir(path):

    pd = os.path.join(path, topic)
    for file in os.listdir(pd):
        pf = os.path.join(pd, file)
        text = open(pf, 'r',encoding="utf-8").read()

        X_valid.append(text)
        Y_valid.append(topic)

#label encode
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)
Y_valid = le.transform(Y_valid)


"""training and tuning"""
name = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe','The gioi', 'The thao', 'Van hoa', 'Vi tinh']
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
save_link = "countvectorizer.txt"
#save_link = "tfidfvectorizer.txt"
report = open(save_link,'w')
accuracy_sc = []
report_ = []

for n in range(1,21,1):
    print("running on min_df = "+str(n))
    for k in range(1,21,1):
        h = 0.05*k
        count_vector = CountVectorizer(min_df=n, max_df=h)
        count_vector.fit(X_train)
        # tfidf_vector = TfidfVectorizer(min_df=n, max_df=h, sublinear_tf=True)
        # tfidf_vector.fit(X_train)

        X_train_count = count_vector.transform(X_train)
        X_valid_count = count_vector.transform(X_valid)
        # X_train_tfidf = tfidf_vector.transform(X_train)
        # X_valid_tfidf = tfidf_vector.transform(X_valid)

        for i in range(1,21,1):
            j = 0.05*i
            time1 = time()
            model = MultinomialNB(alpha=j).fit(X_train_count, Y_train)
            time2 = time()
            y_train_count = model.predict(X_train_count)
            y_valid_count = model.predict(X_valid_count)
            # y_train_tfidf = model.predict(X_train_tfidf)
            # y_valid_tfidf = model.predict(X_valid_tfidf)

            acc_train = accuracy_score(y_train_count, Y_train)
            acc_valid = accuracy_score(y_valid_count, Y_valid)
            # acc_train = accuracy_score(y_train_tfidf, Y_train)
            # acc_valid = accuracy_score(y_valid_tfidf, Y_valid)

            train_time = time2 - time1
            accuracy_sc.append((n,h,j,acc_train,acc_valid, train_time))
            report_.append((classification_report(Y_train, y_train_count, target_names=name),classification_report(Y_valid, y_valid_count, target_names=name)))
            # report_.append((classification_report(Y_train, y_train_tfidf, target_names=name),classification_report(Y_valid, y_valid_tfidf, target_names=name)))

    max_acc_valid = max(accuracy_sc, key=lambda x: x[4])
    report.write(f"* min_df = {max_acc_valid[0]}, max_df = {max_acc_valid[1]}, alpha = {max_acc_valid[2]}, train_acc = {max_acc_valid[3]}, valid_acc = {max_acc_valid[4]}, training_time = {max_acc_valid[5]}\n")
    index = accuracy_sc.index(max_acc_valid)

    report.write("- Train evaluation\n")
    report.write(report_[index][0])
    report.write("- Valid evaluation\n")
    report.write(report_[index][1])
    accuracy_sc.clear()
    report_.clear()

report.close()