import re
import nltk
from gensim.utils import simple_preprocess
from pyvi import ViTokenizer
from nltk.tokenize import sent_tokenize
import os

class Prepro():

    def __init__(self, path_stop_words,path_acronym):
        self.tokenizer = ViTokenizer.tokenize
        self.split_words = simple_preprocess
        self.stop_words = []
        with open(path_stop_words, 'r',encoding="utf-8") as file_stop_words:
            for line in file_stop_words.readlines():
                line = line.replace("\n","",2)
                self.stop_words.append(line)
        self.acronym = []
        with open(path_acronym, 'r', encoding="utf-8") as file_acronym:
            for line in file_acronym.readlines():
                line = line.replace("\n","",2)
                self.acronym.append(line)

    def extract_sent(self, text):
        sentences = sent_tokenize(text)
        return sentences

    def tokenize(self, text):
        return self.tokenizer(text)

    def simple_prepro(self, text):
        words = self.split_words(text)
        sentence = " ".join(words)
        return sentence

    def join_name(self, text):
        words = text.split()
        rs = ""
        uppercase = []
        for w in words:
            if (w[0].isupper()):
                uppercase.append(w)
            else:
                if (len(uppercase) == 0):
                    rs = rs + " " + w
                elif (len(uppercase) == 1):
                    rs = rs + " " + uppercase[0] + " " + w
                    uppercase.clear()
                else:
                    rs = rs + " " + "_".join(uppercase) + " " + w
                    uppercase.clear()
        return rs

    def replace_acronym(self,text):
        words = text.split()
        for w_i in range(len(words)):
            for acro in range(0,len(self.acronym),2):
                if(re.search(self.acronym[acro],words[w_i])):
                    words[w_i] = words[w_i].replace(self.acronym[acro]," "+ self.acronym[acro+1]+ " ",len(words[w_i]))
        return " ".join(words)

    def remove_stops_words(self, text):
        text = " " + text + " "
        for w in self.stop_words:
            text = text.replace(" " + w + " ", " ", len(text))
        return text

    def normalize(self, text):
        rpl_arco = self.replace_acronym(text)
        sent = self.extract_sent(rpl_arco)
        final_sent = ""
        for s in sent:
            join_name = self.join_name(s)
            tokenized = self.tokenize(join_name)
            normal_sen = self.simple_prepro(tokenized)
            rmd_stops = self.remove_stops_words(normal_sen).strip()
            final_sent = final_sent + " " + rmd_stops
        return final_sent.strip()



# link = "../vietnamese_news_ml_dl/10topics/test10_origin/Chinh tri Xa hoi/XH_NLD_T_ (8962).txt"
# link = "../vietnamese_news_ml_dl/10topics/test10_origin/Khoa hoc/KH_NLD_T_ (1931).txt"
# link2 = "../vietnamese_news_ml_dl/vietnamese-stopwords.txt"
p = Prepro("../vietnamese_news_ml_dl/vietnamese-stopwords.txt","../vietnamese_news_ml_dl/tu_viet_tat_update.txt")
# u = "Giám đốc Trung tâm y tế dự phòng huyện Dĩ An (Bình Dương)"
#
# f = open(link, "r",encoding="utf-16").read()
#
# from time import time
# t1 = time()
# # for sent in p.extract_sent(p.replace_acronym(f)):
# #      print(p.remove_stops_words(p.simple_prepro(p.tokenize(p.join_name(sent)))))
# print(p.normalize(f))
# t2 = time()
# print(f"done after: {t2-t1} s")

"""save preprocessed data"""
save_root = '../vietnamese_news_ml_dl/10topics_processed/'
f = "../vietnamese_news_ml_dl/10topics/"
print("Running...")
for folder in os.listdir(f):
  p1 = os.path.join(f, folder)
  p1s = os.path.join(save_root, folder+"_processed")
  try:
    os.mkdir(p1s)
  except:
    pass
  for sub_folder in os.listdir(p1):
    p2 = os.path.join(p1, sub_folder)
    p2s = os.path.join(p1s, sub_folder)
    try:
      os.mkdir(p2s)
    except:
      pass
    for file in os.listdir(p2):
      p3 = os.path.join(p2, file)
      p3s = os.path.join(p2s, file)
      text = open(p3, "r",encoding="utf-16").read()
      text_preprocessed = p.normalize(text)
      with open(p3s, 'w', encoding="utf-8") as save:
        save.write(text_preprocessed)
print("Finished!")