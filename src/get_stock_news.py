import requests
import pprint as pp
from bs4 import BeautifulSoup
import uuid
import sys
from pandas import DataFrame as df
from textblob.classifiers import NaiveBayesClassifier
import codecs


# holds uniquest styles with style text as key and GUID as the value
style_dict = {}
style_count = {}

def get_page_content(url):
    result = None
    page = requests.get(url)
    if(page != None):
        result = page.content

    return result

def get_soup(url):
    soup = None
    content = get_page_content(url)
    soup = BeautifulSoup(content, "lxml")
    return soup

def get_news_headlines(soup):
    news_head_text = []
    news_main_div = soup.find(id="news-main")
    head_a = news_main_div.find_all("a")
    for item in head_a:
        news_head_text.append(item.get_text())
    
    return news_head_text        

def save(headlines, file_name):
    out_df = df(headlines, columns=["headline"])
    out_df = out_df.dropna(how="any")
    out_df.to_csv(file_name)

def get_from_url(googlefin_url):
    soup = get_soup(googlefin_url)
    head_list = get_news_headlines(soup)
    return head_list

#reload(sys)
#sys.setdefaultencoding("utf-8")

#source_url = "https://www.google.com/finance/company_news?q=NYSE%3ATWTR&ei=qHJUWaH7C9Sw2Abq0pfYBw&start=0&num=50"
#soup = get_soup(source_url)
#head_list = get_news_headlines(soup)
#print(head_list)
#save(head_list, "output/twitter_news.csv")
#print(get_from_url(source_url))
