from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
import codecs
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split



def preprocessing(text):

    result = text
    if(isinstance(text, str) or isinstance(text, unicode)):
        #text = text.decode("utf-8", errors="ignore")
        # tokenize into words
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

        # lower capitalization
        tokens = [word.lower() for word in tokens]

        # remove stopwords
        stop = stopwords.words('english')
        stop.extend(["twitter", "twtr", "inc", "inc.", "(", ")", ",", ":", "'s", "n't"])

        tokens = [token for token in tokens if token not in stop]

        # remove words less than three letters
        #tokens = [word for word in tokens if len(word) >= 3]

        # lemmatize
        lmtzr = WordNetLemmatizer()
        tokens = [lmtzr.lemmatize(word) for word in tokens]
        result = ' '.join(tokens)
    return result

def pre_proc_train_file(filename):
    """
    prepare text in train file
    - remove stop words
    - lower caps
    - lemmatize

    expected file structure: first column is sample training text second column is classification
    """
    source_df = pandas.read_csv(filename, encoding="latin1")
    train_data = []
    for row in source_df.itertuples():
        lem_text = preprocessing(row[1])
        train_data.append(  [lem_text, row[2]] )         

    out_df = df(train_data, columns=["text", "classifier"])
    train_df, test_df = train_test_split(out_df, test_size=0.2)

    train_df.to_csv(filename + "scrubbed_train.csv", encoding="latin1")
    test_df.to_csv(filename + "_scrubbed_test.csv", encoding="latin1")
    
    return out_df

def split_unprocessed_file(filename):
    source_df = pandas.read_csv(filename, encoding="latin1")
    train_df, test_df = train_test_split(source_df, test_size=0.2)

    train_df.to_csv(filename + "raw_train.csv", encoding="latin1")
    test_df.to_csv(filename + "_raw_test.csv", encoding="latin1")


def train_nb_v1():
    with codecs.open("output/twitter_news_raw_train.csv", "r", "utf-8", errors="ignore") as fp:
        cl = NaiveBayesClassifier(fp, format="csv")

    
    print(cl.show_informative_features(10))
    test_fp = codecs.open("output/twitter_news_raw_test.csv", "r", "utf-8", errors="ignore")
    print("Accuracy", cl.accuracy(test_fp, format="csv"))
    return cl


def train_nb_v2():
    with codecs.open("output/twitter_news_train.csvscrubbed_train.csv", "r", "utf-8", errors="ignore") as fp:
        cl = NaiveBayesClassifier(fp, format="csv")

    
    print(cl.show_informative_features(10))
    train_fp = codecs.open("output/twitter_news_train.csv_scrubbed_test.csv", "r", "utf-8", errors="ignore")
    print("Accuracy", cl.accuracy(train_fp, format="csv"))
    return cl

def train_dtree_v2():
    with codecs.open("output/twitter_news_train.csv_scrubbed.csv", "r", "utf-8", errors="ignore") as fp:
        cl = DecisionTreeClassifier(fp, format="csv")

    #print(cl.classify(preprocessing("Google stock overvalued")))
    train_fp = codecs.open("output/twitter_news_train.csv_scrubbed_test.csv", "r", "utf-8", errors="ignore")
    print("Accuracy", cl.accuracy(train_fp, format="csv"))
    
    return cl

#pre_proc_train_file("output/twitter_news_train.csv")
#split_unprocessed_file("output/twitter_news_train.csv")
# train_dtree_v2()
#train_nb_v1()
#print(preprocessing("The simple Reason I Still won't Buy Twitter Inc."))