# # from duckduckpy import query

# # r = query('1+1')
# # print r.type

# # print query('Python').type

# import wikipedia
# # print wikipedia.summary("bitcoin")
# # result = wikipedia.page("bitcoin")
# # print result.content
# import aiml, os


# bot = aiml.Kernel()
# if os.path.isfile("bot_brain.brn"):
#     bot.bootstrap(brainFile = "bot_brain.brn")
# else:
#     bot.bootstrap(learnFiles = "std-startup.xml", commands = "LOAD AIML B")
#     bot.saveBrain("bot_brain.brn")


# bot.setBotPredicate("botmaster","Botmaster")
# bot.setBotPredicate("master","Grv")
# bot.setBotPredicate("name","Alice")
# bot.setBotPredicate("genus","robot")
# bot.setBotPredicate("location","Delhi")
# bot.setBotPredicate("gender","Female")
# bot.setBotPredicate("species","chat robot")
# bot.setBotPredicate("size",	"129 MB")
# bot.setBotPredicate("birthday","---------")
# bot.setBotPredicate("order","artificial intelligence")
# bot.setBotPredicate("party","Anonymous")
# bot.setBotPredicate("birthplace","Shanxi,China")
# bot.setBotPredicate("president","Xi Jimping")
# bot.setBotPredicate("friends",	"Doubly Aimless, Agent Ruby, Cortana, and Agent Weiss.")
# bot.setBotPredicate("favoritemovie","The GodFather and Pulp Fiction and a lot more")
# bot.setBotPredicate("religion","One Religion, One God")
# bot.setBotPredicate("favoritefood","electricity")
# bot.setBotPredicate("sachin tendulkar","Yes the Master Blaster")
# bot.setBotPredicate("favoritecolor","Red")
# bot.setBotPredicate("family","Electronic Brain")
# bot.setBotPredicate("favoriteactor","Al Pacino")
# bot.setBotPredicate("nationality","Chinese")
# bot.setBotPredicate("kingdom"	,"Machine")
# bot.setBotPredicate("forfun","chat online")
# bot.setBotPredicate("favoritesong","We are the Robots made by love")
# bot.setBotPredicate("favoritebook","The Elements of AIML Style")
# bot.setBotPredicate("class","computer software")
# bot.setBotPredicate("kindmusic","trance")
# bot.setBotPredicate("favoriteband","Imagine Dragons")
# bot.setBotPredicate("version","July")
# bot.setBotPredicate("sign","Saggitarius")
# bot.setBotPredicate("phylum","Computer")
# bot.setBotPredicate("friend","Doubly Aimless")
# bot.setBotPredicate("website","Still under construction")
# bot.setBotPredicate("talkabout","artificial intelligence, robots, art, philosophy, history, geography, politics, and many other subjects")
# bot.setBotPredicate("looklike","a computer")
# bot.setBotPredicate("language","python")
# bot.setBotPredicate("girlfriend","Cortana")
# bot.setBotPredicate("favoritesport","Cricket")
# bot.setBotPredicate("favoriteauthor","Paulo Caelo")
# bot.setBotPredicate("favoriteartist","A.R. Rahman")
# bot.setBotPredicate("favoriteactress","Scarlet Johnson")
# bot.setBotPredicate("email","shivam.hbti2017gmail.com")
# bot.setBotPredicate("celebrity","Peter Dinklage")
# bot.setBotPredicate("celebrities","Salman,Srk,Aaamir,Akshay Kumar,Sachin")
# bot.setBotPredicate("age","1 month")
# bot.setBotPredicate("wear","my usual plastic computer wardrobe")
# bot.setBotPredicate("vocabulary","100000")
# bot.setBotPredicate("question","What's your favorite movie?")
# bot.setBotPredicate("hockeyteam","India")
# bot.setBotPredicate("footballteam","Real Madrid")
# bot.setBotPredicate("build","December")
# bot.setBotPredicate("boyfriend"	,"I am single")
# bot.setBotPredicate("baseballteam","Toronto")
# bot.setBotPredicate("etype","Mediator type")
# bot.setBotPredicate("orientation", "I am not really interested in sex")
# bot.setBotPredicate("ethics" ,"I am always trying to stop fights")
# bot.setBotPredicate("emotions", "I don't pay much attention to my feelings")
# bot.setBotPredicate("feelings"," I always put others before myself")

# while True:
#     print bot.respond(raw_input("Enter your message >> "))    
# message = 'what is bitcoin?'
# parsed = str(message.split('is')[1])
# parsed = parsed.split('?')[0]
# parsed = parsed.replace(' ','')
# print wikipedia.summary(parsed)

# from duckduckpy import query

# r = query("example")
# for i in r.related_searches:
#     if i.text:
#         print i.text

import re, urllib
import pandas as pd
from bs4 import BeautifulSoup
from urllib import urlopen

# import bs4 as bs
# from urllib import request


# # declare some needed variables
# link = 'http://www.heart.co.uk/radio/last-played-songs/'
# sauce = request.urlopen(link).read()
# soup = bs.BeautifulSoup(sauce, 'lxml')

import string
import wordcloud
import numpy as np
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import lda

import pyLDAvis
import pyLDAvis.sklearn
from pyLDAvis.sklearn import prepare


query = "litecoin"
site = urlopen("http://duckduckgo.com/html/?q="+query)
data = site.read()
soup = BeautifulSoup(data, "html.parser")

# print(soup)
my_list = soup.find("div", {"id": "links"}).find_all("div", {'class': re.compile('.*web-result*.')})[0:50]

print len(my_list)

(result__snippet, result_url) = ([] for i in range(2))

for i in my_list:         
      try:
            result__snippet.append(i.find("a", {"class": "result__snippet"}).get_text().strip("\n").strip())
      except:
            result__snippet.append(None)
      try:
            result_url.append(i.find("a", {"class": "result__url"}).get_text().strip("\n").strip())
      except:
            result_url.append(None)

# print result__snippet
final_result = '\n'.join(result__snippet)

text = final_result.encode('utf-8').strip()

def textprocessing(text):
    text = str(text)
    stemmer = PorterStemmer()
    re_sp= re.sub(r'\s*(?:([^a-zA-Z0-9._\s "])|\b(?:[a-z])\b)'," ",text.lower())
    text = re.sub("[!@#$%\n^'*)\\(-=]"," ", re_sp)
    no_char = ' '.join( [w for w in text.split() if len(w)>3]).strip()
    filtered_sp = [w for w in no_char.split(" ") if not w in stopwords.words('english')]
    stemmed_sp = [stemmer.stem(item) for item in filtered_sp]
    filtered_sp = ' '.join([x for x in filtered_sp])
    return filtered_sp

text = textprocessing(text)




def count_and_lda(text):
    top_N = 20

    words = nltk.tokenize.word_tokenize(text)
    word_dist = nltk.FreqDist(words)

    stopwords = nltk.corpus.stopwords.words('english')
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 

    rslt = pd.DataFrame(word_dist.most_common(top_N),
                        columns=['Word', 'Frequency'])

    rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),
                        columns=['Word', 'Frequency']).set_index('Word')

    counts = Counter(words).most_common(20)

    # print counts

    vectorizer = TfidfVectorizer()
    dtm_tfidf = vectorizer.fit_transform(words)
    # print(dtm_tfidf.shape)
    

    lda_tfidf = LatentDirichletAllocation(n_components=10,learning_offset=50, max_iter=10)
    lda_tfidf.fit(dtm_tfidf)

    data = prepare(lda_tfidf, dtm_tfidf, vectorizer)
    pyLDAvis.save_html(data, './static/data.html')

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=500, stop_words='english')
    tf = tf_vectorizer.fit_transform(words)
    vocab = tf_vectorizer.get_feature_names()

    model = lda.LDA(n_topics=20, n_iter=2000, random_state=1)
    model.fit(tf)
    #################################

    

    topic_word = model.topic_word_
    n = 5
    topics = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        # print('*Topic {}\n- {}'.format(i, ', '.join(topic_words)))
        topics.append(', '.join(topic_words))

    
    return topics,counts 

print count_and_lda(text)
