import requests
import re
from bs4 import BeautifulSoup
import time
import MySQLdb
import nltk
from nltk.corpus import gutenberg
import numpy as np
import emoji
import unicodedata
import pickle
from nltk.corpus import wordnet

from nltk.stem import PorterStemmer
from nltk.stem import RegexpStemmer
from nltk.stem import LancasterStemmer

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
#from emot.emo_unicode import EMOTICONS_EMO
import spacy

from nltk.tokenize.toktok import ToktokTokenizer

import pandas as pd
import numpy as np


EMOTICONS_EMO = {
    ":‑)":"Happy face or smiley",
    ":-))":"Very Happy face or smiley",
    ":-)))":"Very very Happy face or smiley",
    ":)":"Happy face or smiley",
    ":))":"Very Happy face or smiley",
    ":)))":"Very very Happy face or smiley",
    ":-]":"Happy face or smiley",
    ":]":"Happy face or smiley",
    ":-3":"Happy face smiley",
    ":3":"Happy face smiley",
    ":->":"Happy face smiley",
    ":>":"Happy face smiley",
    "8-)":"Happy face smiley",
    ":o)":"Happy face smiley",
    ":-}":"Happy face smiley",
    ":}":"Happy face smiley",
    ":-)":"Happy face smiley",
    ":c)":"Happy face smiley",
    ":^)":"Happy face smiley",
    "=]":"Happy face smiley",
    "=)":"Happy face smiley",
    ":‑D":"Laughing, big grin or laugh with glasses",
    ":D":"Laughing, big grin or laugh with glasses",
    "8‑D":"Laughing, big grin or laugh with glasses",
    "8D":"Laughing, big grin or laugh with glasses",
    "X‑D":"Laughing, big grin or laugh with glasses",
    "XD":"Laughing, big grin or laugh with glasses",
    "=D":"Laughing, big grin or laugh with glasses",
    "=3":"Laughing, big grin or laugh with glasses",
    "B^D":"Laughing, big grin or laugh with glasses",
    ":-))":"Very happy",
    ":-(":"Frown, sad, andry or pouting",
    ":‑(":"Frown, sad, andry or pouting",
    ":(":"Frown, sad, andry or pouting",
    ":‑c":"Frown, sad, andry or pouting",
    ":c":"Frown, sad, andry or pouting",
    ":‑<":"Frown, sad, andry or pouting",
    ":<":"Frown, sad, andry or pouting",
    ":‑[":"Frown, sad, andry or pouting",
    ":[":"Frown, sad, andry or pouting",
    ":-||":"Frown, sad, andry or pouting",
    ">:[":"Frown, sad, andry or pouting",
    ":{":"Frown, sad, andry or pouting",
    ":@":"Frown, sad, andry or pouting",
    ">:(":"Frown, sad, andry or pouting",
    ":'‑(":"Crying",
    ":'(":"Crying",
    ":'‑)":"Tears of happiness",
    ":')":"Tears of happiness",
    "D‑':":"Horror",
    "D:<":"Disgust",
    "D:":"Sadness",
    "D8":"Great dismay",
    "D;":"Great dismay",
    "D=":"Great dismay",
    "DX":"Great dismay",
    ":‑O":"Surprise",
    ":O":"Surprise",
    ":‑o":"Surprise",
    ":o":"Surprise",
    ":-0":"Shock",
    "8‑0":"Yawn",
    ">:O":"Yawn",
    ":-*":"Kiss",
    ":*":"Kiss",
    ":X":"Kiss",
    ";‑)":"Wink or smirk",
    ";)":"Wink or smirk",
    "*-)":"Wink or smirk",
    "*)":"Wink or smirk",
    ";‑]":"Wink or smirk",
    ";]":"Wink or smirk",
    ";^)":"Wink or smirk",
    ":‑,":"Wink or smirk",
    ";D":"Wink or smirk",
    ":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    "X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    "XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    "d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    "=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ">:[(\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":[(\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    "=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    "=[(\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    "=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":‑|":"Straight face",
    ":|":"Straight face",
    ":$":"Embarrassed or blushing",
    ":‑x":"Sealed lips or wearing braces or tongue-tied",
    ":x":"Sealed lips or wearing braces or tongue-tied",
    ":‑#":"Sealed lips or wearing braces or tongue-tied",
    ":#":"Sealed lips or wearing braces or tongue-tied",
    ":‑&":"Sealed lips or wearing braces or tongue-tied",
    ":&":"Sealed lips or wearing braces or tongue-tied",
    "O:‑)":"Angel, saint or innocent",
    "O:)":"Angel, saint or innocent",
    "0:‑3":"Angel, saint or innocent",
    "0:3":"Angel, saint or innocent",
    "0:‑)":"Angel, saint or innocent",
    "0:)":"Angel, saint or innocent",
    ":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    "0;^)":"Angel, saint or innocent",
    ">:‑)":"Evil or devilish",
    ">:)":"Evil or devilish",
    "}:‑)":"Evil or devilish",
    "}:)":"Evil or devilish",
    "3:‑)":"Evil or devilish",
    "3:)":"Evil or devilish",
    ">;)":"Evil or devilish",
    "|;‑)":"Cool",
    "|‑O":"Bored",
    ":‑J":"Tongue-in-cheek",
    "#‑)":"Party all night",
    "%‑)":"Drunk or confused",
    "%)":"Drunk or confused",
    ":-###..":"Being sick",
    ":###..":"Being sick",
    "<:‑|":"Dump",
    "(>_<)":"Troubled",
    "(>_<)>":"Troubled",
    "(';')":"Baby",
    "(^^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(^_^;)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(-_-;)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(~_~;) (・.・;)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(-_-)zzz":"Sleeping",
    "(^_-)":"Wink",
    "((+_+))":"Confused",
    "(+o+)":"Confused",
    "(o|o)":"Ultraman",
    "^_^":"Joyful",
    "(^_^)/":"Joyful",
    "(^O^)／":"Joyful",
    "(^o^)／":"Joyful",
    "(__)":"Kowtow as a sign of respect, or dogeza for apology",
    "_(._.)_":"Kowtow as a sign of respect, or dogeza for apology",
    "<(_ _)>":"Kowtow as a sign of respect, or dogeza for apology",
    "<m(__)m>":"Kowtow as a sign of respect, or dogeza for apology",
    "m(__)m":"Kowtow as a sign of respect, or dogeza for apology",
    "m(_ _)m":"Kowtow as a sign of respect, or dogeza for apology",
    "('_')":"Sad or Crying",
    "(/_;)":"Sad or Crying",
    "(T_T) (;_;)":"Sad or Crying",
    "(;_;":"Sad of Crying",
    "(;_:)":"Sad or Crying",
    "(;O;)":"Sad or Crying",
    "(:_;)":"Sad or Crying",
    "(ToT)":"Sad or Crying",
    ";_;":"Sad or Crying",
    ";-;":"Sad or Crying",
    ";n;":"Sad or Crying",
    ";;":"Sad or Crying",
    "Q.Q":"Sad or Crying",
    "T.T":"Sad or Crying",
    "QQ":"Sad or Crying",
    "Q_Q":"Sad or Crying",
    "(-.-)":"Shame",
    "(-_-)":"Shame",
    "(一一)":"Shame",
    "(；一_一)":"Shame",
    "(=_=)":"Tired",
    "(=^·^=)":"cat",
    "(=^··^=)":"cat",
    "=_^= ":"cat",
    "(..)":"Looking down",
    "(._.)":"Looking down",
    "^m^":"Giggling with hand covering mouth",
    "(・・?":"Confusion",
    "(?_?)":"Confusion",
    ">^_^<":"Normal Laugh",
    "<^!^>":"Normal Laugh",
    "^/^":"Normal Laugh",
    "（*^_^*）" :"Normal Laugh",
    "(^<^) (^.^)":"Normal Laugh",
    "(^^)":"Normal Laugh",
    "(^.^)":"Normal Laugh",
    "(^_^.)":"Normal Laugh",
    "(^_^)":"Normal Laugh",
    "(^^)":"Normal Laugh",
    "(^J^)":"Normal Laugh",
    "(*^.^*)":"Normal Laugh",
    "(^—^）":"Normal Laugh",
    "(#^.^#)":"Normal Laugh",
    "（^—^）":"Waving",
    "(;_;)/~~~":"Waving",
    "(^.^)/~~~":"Waving",
    "(-_-)/~~~ ($··)/~~~":"Waving",
    "(T_T)/~~~":"Waving",
    "(ToT)/~~~":"Waving",
    "(*^0^*)":"Excited",
    "(*_*)":"Amazed",
    "(*_*;":"Amazed",
    "(+_+) (@_@)":"Amazed",
    "(*^^)v":"Laughing,Cheerful",
    "(^_^)v":"Laughing,Cheerful",
    "((d[-_-]b))":"Headphones,Listening to music",
    '(-"-)':"Worried",
    "(ーー;)":"Worried",
    "(^0_0^)":"Eyeglasses",
    "(＾ｖ＾)":"Happy",
    "(＾ｕ＾)":"Happy",
    "(^)o(^)":"Happy",
    "(^O^)":"Happy",
    "(^o^)":"Happy",
    ")^o^(":"Happy",
    ":O o_O":"Surprised",
    "o_0":"Surprised",
    "o.O":"Surpised",
    "(o.o)":"Surprised",
    "o0":"Surprised",
    "(*￣m￣)":"Dissatisfied",
    "(‘A`)":"Snubbed or Deflated"

}

def strip_html_tags(text):
    # text = text.lower()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', text)
    html_index = stripped_text.find('http')
    html_index1 = stripped_text.find('Http')
    html_index2 = stripped_text.find('HTTP')
    while html_index != -1 or html_index1!=-1 or html_index2!=-1:
        # print(stripped_text)
        stripped_text = re.sub(r'http\S+', '', stripped_text)
        stripped_text = re.sub(r'HTTP\S+', '', stripped_text)
        stripped_text = re.sub(r'Http\S+', '', stripped_text)
        html_index = stripped_text.find('http')
        html_index1 = stripped_text.find('Http')
        html_index2 = stripped_text.find('HTTP')
    #print(html_index,html_index1,html_index2)
    #print(stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text=unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode('utf-8','ignore')
    #NFKD or Normalization Form Compatibility Decomposition: Characters are decomposed by compatibility, and multiple combining characters are arranged in a specific order.
    return text

def remove_special_characters(text,remove_digits=False):
    pattern=r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text=re.sub(pattern,'',text)
    return text

def lemmatize_text(text):
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    text=nlp(text)
    text=' '.join([word.lemma_ if word.lemma_!='-PRON-' else word.text for word in text])
    return text

def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text=' '.join([ps.stem(word) for word in text.split()])
    return text

def remove_stopwords(text,is_lower_case=False):
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens=tokenizer.tokenize(text)
    tokens=[token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens=[token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens=[token for token in tokens if token.lower() not in stopword_list]
    filtered_text=' '.join(filtered_tokens)
    return filtered_text


def convert_emoticons(text):
    with open('Emoji_Dict.p', 'rb') as fp:
        Emoji_Dict = pickle.load(fp)
    Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

    for emot in EMOTICONS_EMO:
        #text = re.sub(rr, "_".join(EMOTICONS_EMO[emot].replace(",","").split()), text)
        text = text.replace(emot," ".join(EMOTICONS_EMO[emot].replace(",","").split()))
    return text


def normalize_corpus(corpus,html_stripping=True,contraction_expansion=True,
                     accented_char_removal=True,text_lower_case=True,text_lemmatization=False,
                     special_char_removal=True,stopword_removal=True,remove_digits=True,
                     keep_emo = True):
    normalized_corpus=[]
    for doc in corpus:
        #print('=== raw text\n',doc)

        if html_stripping:
            doc = strip_html_tags(doc)
            #print('=== no html tag\n',doc)

        if keep_emo:
            doc = convert_emoticons(doc)

        if accented_char_removal:
            doc=remove_accented_chars(doc)
            #print('=== no accented chars\n',doc)
        if text_lower_case:
            doc=doc.lower()
            #print('=== lower case\n',doc)
        # remove extra newlines
        doc=re.sub(r'[\r|\n|\r\n]+',' ',doc)
        if text_lemmatization:
            doc=lemmatize_text(doc)
            #print('=== lemmas\n',doc)
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern=re.compile(r'([{.(-)!}])')
            doc=special_char_pattern.sub(" \\1 ",doc)
            doc=remove_special_characters(doc,remove_digits=remove_digits)
            #print('=== no special chars\n',doc)
        # remove extra whitespace
        doc=re.sub(' +',' ',doc)
        if stopword_removal:
            doc=remove_stopwords(doc,is_lower_case=text_lower_case)
            #print('=== no stopwords\n',doc)
        normalized_corpus.append(doc)
    return normalized_corpus

def get_text_movies(data,type):
    movie_data = data.groupby(['imdb_id'])
    movies = []
    texts = []
    if type==0:
        column_name = 'message_and_description' #post
    elif type == 1:
        column_name = 'message' #comment
    else:
        column_name = ['review_header','review_content'] #review

    for movie, mdf in movie_data:

        text = ''
        space = '.'
        movies.append(movie)
        try:
            if type == 2:

                for i in range(len(mdf)):
                    text += list(mdf['review_header'])[i] + space + list(mdf['review_content'])[i] + space
            else:
                mdf[column_name] = [str(x) for x in list(mdf[column_name])]
                text = space.join(list(mdf[column_name]))
            texts.append(text.replace('\n', ''))

        except Exception as e:
            if type == 2:
                mdf['review_header'] = [str(x) for x in list(mdf['review_header'])]
                mdf['review_content'] = [str(x) for x in list(mdf['review_content'])]
                for i in range(len(mdf)):
                    text += list(mdf['review_header'])[i] + space + list(mdf['review_content'])[i] + space
            else:
                mdf[column_name] = [str(x) for x in list(mdf[column_name])]
                text = space.join(list(mdf[column_name]))
            texts.append(text.replace('\n', ''))

    movie_texts = pd.DataFrame()
    movie_texts['imdb_id'] = movies
    movie_texts['text'] = texts
    return movie_texts


def use_text(data,type):
    new_data = pd.DataFrame()
    new_data['imdb_id'] = data['imdb_id']
    if type==0:
        try:
            data['message_and_description'] = normalize_corpus(list(data['message_and_description']))
        except Exception as e:
            data['message_and_description'] = [str(x) for x in list(data['message_and_description'])]
            data['message_and_description'] = normalize_corpus(list(data['message_and_description']))
        new_data['posttext'] = data['message_and_description']

    elif type==1:
        try:
            data['message'] = normalize_corpus(list(data['message']))
        except Exception as e:
            data['message'] = [str(x) for x in list(data['message'])]
            data['message'] = normalize_corpus(list(data['message']))
        new_data['comment_text'] = data['message']

    else:
        try:
            data['review_header']=normalize_corpus(list(data['review_header']))
            data['review_header']=normalize_corpus(list(data['review_header']))
        except Exception as e:
            data['review_header'] = [str(x) for x in list(data['review_header'])]
            data['review_content'] = [str(x) for x in list(data['review_content'])]
            data['review_header'] = normalize_corpus(list(data['review_header']))
            data['review_content'] = normalize_corpus(list(data['review_content']))
        new_data['review_header'] = data['review_header']
        new_data['review_content'] = data['review_content']
    return new_data



if __name__ == "__main__":

    post_file = 'fbposts.csv'
    post_textfile = pd.read_csv(post_file)

    comment_file = 'fbcomments.csv'
    comment_textfile = pd.read_csv(comment_file)

    review_file = 'fandango_review.csv'
    review_textfile = pd.read_csv(review_file)

    all_file = [post_textfile,comment_textfile,review_textfile]
    all_textdata = []

    for i, file in enumerate(all_file):
        all_textdata.append(use_text(file, i))

    for i, file in enumerate(all_textdata):
        if i==0:
            file.to_csv('post.csv', index = False)
        elif i==1:
            file.to_csv('comment.csv', index=False)
        else:
            file.to_csv('review.csv',index = False)

