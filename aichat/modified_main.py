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
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
#from mpld3.display import display_d3
import mpld3


WORD = re.compile(r'\w+')

app = Flask(__name__, static_url_path='')

@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/howto")
def howto():
    return render_template('howto.html')
@app.route("/contact")
def contact():
    return render_template('contact.html')


bot = aiml.Kernel()
if os.path.isfile("bot_brain.brn"):
    bot.bootstrap(brainFile = "bot_brain.brn")
else:
    bot.bootstrap(learnFiles = "aiml/std-startup.xml", commands = "LOAD AIML B")
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
    words_except_stopwords = [w for w in words if w not in stopwords and len(w)>2]
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
    segrated_result['Text Relevance'] = segrated_result['Text Relevance'].astype('float64')
    segrated_result = segrated_result[segrated_result['Text Relevance'] > 30]
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
from Queue import Queue
from threading import Thread
def scrape_and_parse_modified(query):
    #print query
    site = urlopen("http://duckduckgo.com/html/?q="+query)
    data = site.read()
    soup = BeautifulSoup(data, "html.parser")

    my_list = soup.find_all("a", {"class": "result__url"})
    links = [a['href'] for a in my_list if a.has_key('href')]
    result_url = []
    for link in links:
        split_link = link.split("&uddg=")
        if "youtube" in split_link[1]:
            continue
        result_url.append(split_link[1].encode('utf-8'))

    #print(result_url)
    # Dataframe of results with two columns url, snippet this dataframe will be further be used in predicting Latent Dirichlet Allocation(lda)
    
    urls = []
    snippets = []
    q=Queue()
    def worker():
        while True:
            item=q.get()
            try :
                site = urlopen(urllib.unquote(item))
                data = site.read()
                soup = BeautifulSoup(data, "html.parser")
                html_tags = [ 'p',]
                snippet = ''
                for html_tag in html_tags:
                    for tag_data in soup.find_all(html_tag):
                        snippet =  snippet + tag_data.text + ' '
                    print item
                if len(snippet) > 100:
                    snippets.append(snippet.encode('utf-8'))
                    urls.append(item.encode('utf-8'))
            except Exception as e:
                print e
                print(item, 'cannote be parsed')

            #all_results.append(item[0])
            q.task_done()
    for s in result_url:
        q.put(s)
    workers=30
    for i in range(workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()
    # for url in result_url:
    #     try :
    #         site = urlopen(urllib.unquote(url))
    #         data = site.read()
    #         soup = BeautifulSoup(data, "html.parser")
    #         html_tags = [ 'p',]
    #         snippet = ''
    #         for html_tag in html_tags:
    #             for tag_data in soup.find_all(html_tag):
    #                 snippet =  snippet + tag_data.text + ' '
    #             print snippet
    #         if len(snippet) > 100:
    #             snippets.append(snippet.encode('utf-8'))
    #             urls.append(url.encode('utf-8'))
    #     except Exception:
    #         print(url, 'cannote be parsed')
    segrated_result = pd.DataFrame(
        {
            'url': urls,
            'snippet': snippets
        }
    )

    return segrated_result

def display_topics(model, feature_names, no_top_words):
    #topic_idx = 0
    #topic_list = []
    display_topics = []
    #topic_list = []
    for topic_idx, topic in enumerate(model.components_):
        #print "Topic %d:" % (topic_idx)
        #print ",".join([feature_names[i]for i in topic.argsort()[:-no_top_words - 1:-1]][1:])
        topic = " ".join([feature_names[i]for i in topic.argsort()[:-no_top_words - 1:-1]])
        
        #topic_list.append([feature_names[i]for i in topic.argsort()[:-no_top_words - 1:-1]])
        #topic_list.append( "Topic "+str(topic_idx)+":")
        #topic_list.append(topic)
        display_topics.append(topic)
        
    return display_topics
        

def topic_sim(segrated_result): 
    #print segrated_result.info
    texts_list = segrated_result.snippet2.tolist()

    #print(texts_list)
    vectorizer = TfidfVectorizer()
    dtm = vectorizer.fit_transform(texts_list)
    invtrm=vectorizer.inverse_transform(dtm)
    #print invtrm
    #print(invtrm)
    vocab = vectorizer.get_feature_names()
    #print(vectorizer.vocabulary_)
    #print dtm.shape
    
    scipy.sparse.csr.csr_matrix
    dtm = dtm.toarray()  # convert to a regular array
    #print(dtm)
    #vocab = np.array(vocab)
    #print(vocab)
    # for v in vocab:
    #     print(v)
    dist = 1 - cosine_similarity(dtm)
    np.round(dist, 2)
    np.round(dist, 2).shape
    # norms = np.sqrt(np.sum(dtm * dtm, axis=1, keepdims=True))  # multiplication between arrays is element-wise
    # dtm_normed = dtm / norms
    # similarities = np.dot(dtm_normed, dtm_normed.T)
    # np.round(similarities, 2)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    #print dist
    xs, ys = pos[:, 0], pos[:, 1]
    names = list(reversed(range(1,len(xs),1)))
    # plt.style.use('fivethirtyeight')
    
    # for x, y, name in zip(xs, ys, names):
    #     plt.title("Visualizing distances between the different text corpuses")
    #     plt.scatter(x, y)
    #     plt.text(x, y, name)
    # #plt.show()
    
#     fig, ax = plt.subplots()
#     np.random.seed(0)
#     color, size = np.random.random((2, len(xs)))
#     for x, y, name in zip(xs, ys, names):
# #         #ax.plot(np.random.normal(size=100),np.random.normal(size=100),'or', ms=10, alpha=0.3)
# #         #ax.plot(np.random.normal(size=100),np.random.normal(size=100),'ob', ms=20, alpha=0.1)
# #         
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_title('Visualizing distances between the different text corpuses', size=15)
#         ax.grid(color='lightgray', alpha=0.7)
# #         #ax.plot(x, y)
# #         
#         ax.scatter(x, y, c=color, s=500 * size, alpha=0.3)
#         ax.text(x,y,name)

    fig, ax = plt.subplots()
    #N = 100
    color, size = np.random.random((2, len(xs)))
    scatter = ax.scatter(xs,
                         ys,
                         c=color,
                         s=1000 * size,
                         alpha=0.3,
                         cmap=plt.cm.jet)
    ax.grid(color='lightgray', linestyle='solid', alpha=0.7)
    
    ax.set_title('Scatter plot of text corpuses distances', size=16)
    
    labels = ['Text {0}'.format(i) for i in names]
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)
    
   

    
#     # Scatter points
#     fig, ax = plt.subplots()
#     np.random.seed(0)
#     #x, y = np.random.normal(size=(2, 200))
#     color, size = np.random.random((2, len(xs)))
#     print names
#     ax.scatter(xs, ys, c=color, s=500 * size, alpha=0.3)
#     #ax.text(xs,ys,names)
#     ax.grid(color='lightgray', alpha=0.7)
        
    
    tfidf_feature_names = vectorizer.get_feature_names()
    no_topics = 20
    # Run NMF
    #nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(dtm)
    # Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=500, learning_method='online', learning_offset=50.,random_state=0).fit(dtm)
    no_top_words = 10
    #display_topics(nmf, tfidf_feature_names, no_top_words)
    #display_topics(lda, tf_feature_names, no_top_words)
    display_result = display_topics(lda, tfidf_feature_names, no_top_words)

    #return mpld3.display(fig)
    return  display_result, mpld3.display()


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
    segrated_result=[]
    topic_result=[]
    #print(result)
    display_result=[]
    query_result=result
    # Here i have modified the added ->or "less context" in result or "too complex" in result
    if ("I do not know" in result or "less context" in result or "too complex" in result):
        # parsed = str(answer.split('is')[0])
        # parsed = parsed.split('?')[0]
        print(result)
        print(answer)
        answer=answer.lower()
        if "I do not know" in result:
            parsed= result.split('I do not know')
            parsed=parsed[1]
            parsed=parsed.split('.')
            parsed=parsed[0]
            parsed=parsed.split(' ')
            wiki_search=' '.join(parsed[1:-1])
            parsed=parsed[0]+''.join(parsed[-1:])+' '.join(parsed[1:-1])+'?'
        else:
            parsed = str(answer.split('what')[1])
            parsed = parsed.split('is')[1]
            wiki_search=parsed
            parsed = "what is" + parsed + "?"
        # parsed = parsed.replace(' ','')
        # result = wikipedia.summary(parsed)
        segrated_result = scrape_and_parse_modified(parsed)
        
        try:
            # To represent in the chatbox
            # Crating try and except to either use wiki history or the first text of segregated dataframe (which is the previous logic) 
            query_result = wikipedia.summary(wiki_search)
        except:
            query_result = result = result[0][0:300] + "...." + "\nIf you were looking for some other query, please ask in more detailed manner or refer to the charts and links displayed."
         
         
         
        segrated_result['Text Relevance'] = segrated_result['snippet'].apply(lambda text: get_cosine(query_result, text))*100
        
        
        # print result
        segrated_result['snippet2']=segrated_result['snippet']
        segrated_result['snippet'] = segrated_result['snippet'].apply(lambda text:textprocessing(text))
        result = segrated_result['snippet'].tolist()
        result = result[0][0:300] + "...."
        segrated_result=segrated_result.sort_values(by='Text Relevance', ascending=1)
        
        segrated_result, hex_data = hex_map_data(segrated_result)
        segrated_result['Text'] = list(reversed(range(1,len(segrated_result)+1) ))
        #hex_data=hex_data.sort_values(by='Text Relevance', ascending=0)
        display_result,topic_result = topic_sim(segrated_result)
        #print(hex_data)
        segrated_result=segrated_result.to_dict('records')
        


    all_data = {'result':query_result, 'hex_data': hex_data,'segregated':segrated_result,
            'topic_result':topic_result,'display_result':display_result}
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
