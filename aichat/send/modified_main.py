from flask import Flask, render_template, request, jsonify, send_from_directory
import aiml
import os
from duckduckpy import query
import wikipedia
import re, urllib, math
import pandas as pd
from bs4 import BeautifulSoup
from urllib import urlopen
import string
import wordcloud
import numpy as np
from wordcloud import WordCloud, STOPWORDS 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import lda
import pyLDAvis
import pyLDAvis.sklearn
from pyLDAvis.sklearn import prepare
from collections import Counter
import requests

WORD = re.compile(r'\w+')

app = Flask(__name__, static_url_path='')

@app.route("/")
def hello():
    return render_template('chat.html')


bot = aiml.Kernel()
if os.path.isfile("bot_brain.brn"):
    bot.bootstrap(brainFile = "bot_brain.brn")
else:
    bot.bootstrap(learnFiles = "std-startup.xml", commands = "LOAD AIML B")
    bot.saveBrain("bot_brain.brn")

bot.setBotPredicate("botmaster","Botmaster")
bot.setBotPredicate("master","Grv")
bot.setBotPredicate("name","Alice")
bot.setBotPredicate("genus","robot")
bot.setBotPredicate("location","Delhi")
bot.setBotPredicate("gender","Female")
bot.setBotPredicate("species","chat robot")
bot.setBotPredicate("size",	"129 MB")
bot.setBotPredicate("birthday","---------")
bot.setBotPredicate("order","artificial intelligence")
bot.setBotPredicate("party","Anonymous")
bot.setBotPredicate("birthplace","Shanxi,China")
bot.setBotPredicate("president","Xi Jimping")
bot.setBotPredicate("friends",	"Doubly Aimless, Agent Ruby, Cortana, and Agent Weiss.")
bot.setBotPredicate("favoritemovie","The GodFather and Pulp Fiction and a lot more")
bot.setBotPredicate("religion","One Religion, One God")
bot.setBotPredicate("favoritefood","electricity")
bot.setBotPredicate("sachin tendulkar","Yes the Master Blaster")
bot.setBotPredicate("favoritecolor","Red")
bot.setBotPredicate("family","Electronic Brain")
bot.setBotPredicate("favoriteactor","Al Pacino")
bot.setBotPredicate("nationality","Chinese")
bot.setBotPredicate("kingdom"	,"Machine")
bot.setBotPredicate("forfun","chat online")
bot.setBotPredicate("favoritesong","We are the Robots made by love")
bot.setBotPredicate("favoritebook","The Elements of AIML Style")
bot.setBotPredicate("class","computer software")
bot.setBotPredicate("kindmusic","trance")
bot.setBotPredicate("favoriteband","Imagine Dragons")
bot.setBotPredicate("version","July")
bot.setBotPredicate("sign","Saggitarius")
bot.setBotPredicate("phylum","Computer")
bot.setBotPredicate("friend","Doubly Aimless")
bot.setBotPredicate("website","Still under construction")
bot.setBotPredicate("talkabout","artificial intelligence, robots, art, philosophy, history, geography, politics, and many other subjects")
bot.setBotPredicate("looklike","a computer")
bot.setBotPredicate("language","python")
bot.setBotPredicate("girlfriend","Cortana")
bot.setBotPredicate("favoritesport","Cricket")
bot.setBotPredicate("favoriteauthor","Paulo Caelo")
bot.setBotPredicate("favoriteartist","A.R. Rahman")
bot.setBotPredicate("favoriteactress","Scarlet Johnson")
bot.setBotPredicate("email","shivam.hbti2017gmail.com")
bot.setBotPredicate("celebrity","Peter Dinklage")
bot.setBotPredicate("celebrities","Salman,Srk,Aaamir,Akshay Kumar,Sachin")
bot.setBotPredicate("age","1 month")
bot.setBotPredicate("wear","my usual plastic computer wardrobe")
bot.setBotPredicate("vocabulary","100000")
bot.setBotPredicate("question","What's your favorite movie?")
bot.setBotPredicate("hockeyteam","India")
bot.setBotPredicate("footballteam","Real Madrid")
bot.setBotPredicate("build","December")
bot.setBotPredicate("boyfriend"	,"I am single")
bot.setBotPredicate("baseballteam","Toronto")
bot.setBotPredicate("etype","Mediator type")
bot.setBotPredicate("orientation", "I am not really interested in sex")
bot.setBotPredicate("ethics" ,"I am always trying to stop fights")
bot.setBotPredicate("emotions", "I don't pay much attention to my feelings")
bot.setBotPredicate("feelings"," I always put others before myself")

# Stage two codes goes here 
# in this what iam doing is finding cosine distance and then sending coisine distance in form hex map to user
def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def hex_map_labels(text):
    words = nltk.tokenize.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    words_except_stopwords = [w for w in words if w not in stopwords]
    counts = Counter(words_except_stopwords).most_common(5)
    lable_list = []
    for count in counts:
        lable = {
                'lable': count[0],
                'count': count[1]
            }
        lable_list.append(lable)
    return lable_list


def hex_map_data(segrated_result):
    master_text = segrated_result['snippet'][0]
    segrated_result['cosine_distance'] = segrated_result['snippet'].apply(lambda text: get_cosine(master_text, text))
    segrated_result['lables'] = segrated_result['snippet'].apply(lambda text: hex_map_labels(text))
    output_dict = segrated_result.to_dict('records')
    return segrated_result, output_dict


def textprocessing(text):
    text = str(text)
    stemmer = PorterStemmer()
    text.replace('`', "")
    text.replace("\"", "")
    re_sp= re.sub(r'\s*(?:([^a-zA-Z0-9._\s "])|\b(?:[a-z])\b)'," ",text.lower())
    text = re.sub("[!@#$%\n^'*)\\(-=]"," ", re_sp)
    no_char = ' '.join( [w for w in text.split() if len(w)>3]).strip()
    filtered_sp = [w for w in no_char.split(" ") if not w in stopwords.words('english')]
    stemmed_sp = [stemmer.stem(item) for item in filtered_sp]
    filtered_sp = ' '.join([x for x in filtered_sp])
    return filtered_sp

#  This is stage 3 code nothing is modified in it
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

    topic_word = model.topic_word_
    n = 5
    topics = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        # print('*Topic {}\n- {}'.format(i, ', '.join(topic_words)))
        topics.append(', '.join(topic_words))

    the_counts = []
    for count in counts:
        the_counts.append({'data':count[0], 'value':count[1]})
        

    return topics,the_counts

def scrape_and_parse_modified(query):
    site = urlopen("http://duckduckgo.com/html/?q="+query)
    data = site.read()
    soup = BeautifulSoup(data, "html.parser")

    my_list = soup.find_all("a", {"class": "result__url"})
    links = [a['href'] for a in my_list if a.has_key('href')]
    result_url = []
    for link in links:
        split_link = link.split("&uddg=")
        result_url.append(split_link[1].encode('utf-8'))

    print(result_url)
    # Dataframe of results with two columns url, snippet this dataframe will be further be used in predicting Latent Dirichlet Allocation(lda)
    
    urls = []
    snippets = []

    
    for url in result_url:
        try :
            site = urlopen(urllib.unquote(url))
            data = site.read()
            soup = BeautifulSoup(data, "html.parser")
            html_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a']
            snippet = ''
            for html_tag in html_tags:
                for tag_data in soup.find_all(html_tag):
                    snippet =  snippet + tag_data.text + ' '
                print snippet
            if len(snippet) > 100:
                snippets.append(snippet.encode('utf-8'))
                urls.append(url.encode('utf-8'))
        except Exception:
            print(url, 'cannote be parsed')
    segrated_result = pd.DataFrame(
        {
            'url': urls,
            'snippet': snippets
        }
    )

    return segrated_result



def parse_message(answer):
    if("..." in answer):
        result = answer + "wait! Are you testing me?!?!"
    elif('http' in answer):
        result = "TBH i don't know...but i can give you a link....go find it there :)\n"+answer
    elif(answer == "save"):
        bot.saveBrain("bot_brain.brn")
        result = 'It is saved!'
    else:
        result = bot.respond(answer)
    
    topics = []
    the_counts = []
    hex_data = []
    print(result)
    # Here i have modified the added ->or "less context" in result or "too complex" in result
    if ("I do not know" in result or "less context" in result or "too complex" in result):
        parsed = str(answer.split('is')[0])
        parsed = parsed.split('?')[0]
        # parsed = parsed.replace(' ','')
        # result = wikipedia.summary(parsed)
        segrated_result = scrape_and_parse_modified(parsed)

        # print result
        segrated_result['snippet'] = segrated_result['snippet'].apply(lambda text:textprocessing(text))
        result = segrated_result['snippet'].tolist()
        result = result[0][0:300] + "...."
        segrated_result, hex_data = hex_map_data(segrated_result)
        print(hex_data)

    all_data = {'result':result, 'hex_data': hex_data}
    return all_data

@app.route('/ask', methods=['GET', 'POST'])
def parse_request():
    message =  str(request.form['message'])
    all_data = parse_message(message)
    # query_result = wikipedia.summary(parsed)
    return jsonify({ 'answer': all_data})

@app.route('/lda', methods=['GET', 'POST'])
def lda_request():
    text =  str(request.form['text'])
    topics = []
    the_counts = []
    topics, the_counts= count_and_lda(text)
    all_data = {'topics':topics, 'the_counts': the_counts}
    return jsonify({ 'answer': all_data})

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=False)    
