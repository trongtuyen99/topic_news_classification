import os
# import nltk
# nltk.download('punkt')
from topic_news_classification.prepro_code.preprocess import Prepro
PATH_STOPSWORD = r"../../data/vietnamese_stopsword"
PATH_ACRONYM = r"../../data/vietnamese_tvt"
prepro = Prepro(PATH_STOPSWORD, PATH_ACRONYM)

data = "ngày 11-12-1999 3 thanh niên đi xe Việt Nam giải phóng#@$##$@"
out = prepro.normalize(data)
print(out)

# print(os.listdir("../../data"))
