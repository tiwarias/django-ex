from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
import os, requests, uuid, json
import pandas as pd
import numpy as np
from io import BytesIO,StringIO
import xlsxwriter
import spacy
nlp = spacy.load('en_core_web_lg')
from spacy_langdetect import LanguageDetector
nlp.add_pipe(LanguageDetector(),name="LanguageDetector",last=True)
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from spacy.lemmatizer import Lemmatizer
import pickle, joblib
import string
import re
import matplotlib.pyplot as plt 
from collections import Counter
from PIL import Image
from sklearn.svm import LinearSVC
import configparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
analyzer_vader = SentimentIntensityAnalyzer()

# remove words from stop words list

nlp.vocab["not"].is_stop = False
nlp.vocab["never"].is_stop = False
nlp.vocab['im'].is_stop = True
nlp.vocab["ive"].is_stop = True
nlp.vocab["nothing"].is_stop = False

class classification_class:
    
    def __init__(self):
        
        self.df_agent = pd.read_csv("./BI_Chat_Analysis/agents.csv")
        self.df_overall = pd.read_csv("./BI_Chat_Analysis/overall.csv")
        self.df_similar = pd.read_csv("C:/Labs/BI_All_Channels/BI_Chat_Analysis/BI_Chat_Analysis/Dataset.csv")
        self.df_similar['keyword_NoPunct'] = self.df_similar['Keyword'].apply(lambda x: self.remove_punct(x))
        self.df_similar['keyword_tokenized'] = self.df_similar['keyword_NoPunct'].apply(lambda x: self.tokenization(x.lower()))
        self.df_similar['keywordRemoveStopWords'] = self.df_similar['keyword_tokenized'].apply(lambda x: self.remove_stopwords(x))
        self.df_similar['keyword_nostop'] = self.df_similar['keywordRemoveStopWords'].apply(lambda x: self.convert_string(x))

        self.df_similar['keyword_lemmas'] = self.df_similar['keyword_nostop'].apply(lambda x: self.lemmatization(x))

        self.df_similar['String_keyword'] = self.df_similar['keyword_lemmas'].apply(lambda x: self.convert_string(x))
        self.df_similar['String_nlp'] = self.df_similar['String_keyword'].apply(lambda x: nlp(x))
        
        
    
    def remove_punct(self,text):
        text  = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', ' ', text)
        return text
    
    def tokenization(self,text):
        text = nlp(text)
        text = [token.text for token in text]
        return text
    
    def remove_stopwords(self,text):
        filtered_sentence = []
        for word in text:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word)
        return filtered_sentence
    
    def lemmatization(self,text):
        text = nlp(text)
        text_lemma = [token.lemma_ for token in text]
        return text_lemma
    
    def convert_string(self,list_value):
        return ' '.join(list_value)
    
    def get_categories(self,doc):
        pred_similar = ''
        ratio_list = []
        flag_agent = 0
        for i in self.df_agent['agents']:
            if i in doc:
                pred_similar = self.df_agent['Category'].unique()[0]
                flag_agent = 1
                break
        if flag_agent == False:
            nlp_doc = nlp(doc)
            for token in nlp_doc:
                if token.pos_ == "PROPN" and token.tag_ == "NNP" and token.ent_type_ == "PERSON":
                    pred_similar = self.df_agent['Category'].unique()[0]
                    flag_agent = 1
                    break
        if flag_agent == False:
            flag_overall = any(element in doc for element in self.df_overall['keyword'].values)
            if flag_overall == False:
                pred_similar = self.df_overall['Category'].unique()[0]
                flag_agent = 1
        
        if flag_agent == False:
            nlp_doc = nlp(doc)
            ratio_list = list(map(lambda x: nlp_doc.similarity(x),self.df_similar['String_nlp'].values))
            pred_index = ratio_list.index(max(ratio_list))
            pred_similar = self.df_similar.loc[pred_index,'Category']
        return pred_similar

    def classify(self,data):
        data['Comments_NoPunct'] = data['Translated_Comments'].apply(lambda x: self.remove_punct(x))
        data['Comments_tokenized'] = data['Comments_NoPunct'].apply(lambda x: self.tokenization(x.lower()))
        data['CommentsRemoveStopWords'] = data['Comments_tokenized'].apply(lambda x: self.remove_stopwords(x))
        data['string_nostop'] = data['CommentsRemoveStopWords'].apply(lambda x: self.convert_string(x))
        data['lemmas'] = data['string_nostop'].apply(lambda x: self.lemmatization(x))
        data['String_Comments'] = data['lemmas'].apply(lambda x: self.convert_string(x))
        pred_list = []
        sentiment_list = []
        
        pred_list = list(map(lambda x :self.get_categories(x),data['String_Comments'].values))
        sentiment_list = list(map(lambda y :self.sentiments_prediction(y),data['Translated_Comments'].values))
        
        return pred_list,sentiment_list

    def sentiment_validation(self,comment):
        blob = TextBlob(comment,analyzer=NaiveBayesAnalyzer())
        val = blob.sentiment[0]
        if val == 'pos':
            sentiment_value = 'Positive'
        else:
            sentiment_value = 'Negative'
        return sentiment_value    
    
    def sentiments_prediction(self,text):
        
        score = analyzer_vader.polarity_scores(text)
        sentiment_val = ''
        if score['neu'] == 1.0:
            sentiment_val = self.sentiment_validation(text)
        elif score['compound'] > 0.25:
            sentiment_val= 'Positive'
        
        elif 0.25 >= score['compound'] >= 0.01:
            if score['neu'] > 0.83:
                sentiment_val = self.sentiment_validation(text)
            else:
                sentiment_val= 'Positive'
        elif score['compound'] <= -0.5:
            sentiment_val = 'Negative'
        elif 0.0 > score['compound'] > -0.5:
            if score['pos'] == 0.0:
                sentiment_val = 'Negative'
            else:
                sentiment_val = self.sentiment_validation(text)
        else:
            sentiment_val = self.sentiment_validation(text)  
        
        return sentiment_val

@method_decorator(login_required, name = 'graphs')
class translate_class:
     
    df = pd.DataFrame()
    
    def __init__(self):
        
        self.obj_classify_comm = main_classification()
    
    
    # Create your views here.
    def Home(self,request):
        render(request, 'index.html')
    
    def display_wordcloud(self,df_cloud):
        df_cloud.dropna(inplace=True)
        df_cloud.reset_index(drop=True, inplace=True)
    
        file1 = open(r"./BI_Chat_Analysis/myfile.txt", "w", encoding="utf-8") 
        for i in range(len(df_cloud)):
            file1 = open(r"./BI_Chat_Analysis/myfile.txt", "a", encoding="utf-8") 
            x = df_cloud.loc[i,'Translated_Comments']
            file1.write(x) 
        file1.close()
    
        
        doc=nlp(open(r"./BI_Chat_Analysis/myfile.txt",encoding="latin-1").read())
        
        words = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        words = [word.lower() for word in words]
        unwanted = {'chatbot', 'bot', 'chat', 'phone' , 'hear', 'process', 'wait','time'}
        words = [ele for ele in words if ele not in unwanted]
        
        word_freq = Counter(words)
        common_words = word_freq.most_common(50)
        
        ''' Convert to dict to be read by j3'''    
    
        list_word = []
        for i in range(len(common_words)):
            key = common_words[i][0]
            val = common_words[i][1]
            list_word.append({'word' : key , 'size' : val})
        
        return list_word
    
    def graphs(self,request):
        
        contact_type = translate_class.df['Contact Type'].unique()
        
        total = []
        positive =[]
        negative = []
        
        for i in range(len(contact_type)):
            temp_df = translate_class.df.loc[translate_class.df['Contact Type'] == contact_type[i]]
            total.append(temp_df.shape[0])
            positive.append(temp_df.loc[temp_df['Sentiments']=='Positive'].shape[0])
            negative.append(temp_df.loc[temp_df['Sentiments']=='Negative'].shape[0])
        
        c_type = contact_type.tolist()
    
        return render(request,"dashboard.html", {'total':total,'positive':positive,'negative' :negative})


    
    def graphs_contact_type(self,request):
        if request.method == "POST":
            contact_type = request.POST.get('channel_input')

            if contact_type:
                filtered_df = translate_class.df[(translate_class.df['Contact Type'] == contact_type)]
                
                satisfaction_values = filtered_df['Satisfaction Level'].value_counts().to_dict()
                category_values = filtered_df['Category'].value_counts().to_dict()
                sentiment_values = filtered_df['Sentiments'].value_counts().to_dict()
            
                satisfaction_list = list(satisfaction_values.values())
                category_list =  list(category_values.values())
                sentiments_list = list(sentiment_values.values())   
            
                satisfaction_keys = list(satisfaction_values.keys())
                category_keys = list(category_values.keys())
                sentiment_keys = list(sentiment_values.keys())
            
            
                total_category = sum(category_list)
                df_wordcloud = filtered_df[['Translated_Comments']]
                wordcloud = self.display_wordcloud(df_wordcloud)

                return render(request,"channel_wise.html", {'sent_data':sentiments_list,'sent_keys':sentiment_keys,'sat_data' :satisfaction_list,'cat_keys_graph':category_keys,'sat_keys' :satisfaction_keys, 'cat_data':category_list, 'category_sum' : total_category,'wordcloud' : wordcloud})
    
        satisfaction_values = translate_class.df['Satisfaction Level'].value_counts().to_dict()
        category_values = translate_class.df['Category'].value_counts().to_dict()
        sentiment_values = translate_class.df['Sentiments'].value_counts().to_dict()
    
        satisfaction_list = list(satisfaction_values.values())
        category_list =  list(category_values.values())
        sentiments_list = list(sentiment_values.values())   
    
        satisfaction_keys = list(satisfaction_values.keys())
        category_keys = list(category_values.keys())
        sentiment_keys = list(sentiment_values.keys())
    
    
        total_category = sum(category_list)
        df_wordcloud = translate_class.df[['Translated_Comments']]
        wordcloud = self.display_wordcloud(df_wordcloud)
    
        return render(request,"channel_wise.html", {'sent_data':sentiments_list,'sent_keys':sentiment_keys,'sat_data' :satisfaction_list,'cat_keys_graph':category_keys,'sat_keys' :satisfaction_keys, 'cat_data':category_list, 'category_sum' : total_category,'wordcloud' : wordcloud})


    def detect_lang(self,doc):
        doc = nlp(doc)
        lang = doc._.language['language']
        return lang


    def translate_comments(self,request):
        if request.method == "POST":
            comment_file =  request.FILES.get('myfile')
            
            if comment_file.name.endswith('.csv'):
                df1 = pd.read_csv(comment_file,sheet_name='Survey')
                df2 = pd.read_csv(comment_file,sheet_name='Comments')
            elif comment_file.name.endswith('.xls'):
                df1 = pd.read_excel(comment_file,sheet_name='Survey')
                df2 = pd.read_excel(comment_file,sheet_name='Comments')
            elif comment_file.name.endswith('.xlsx'):
                df1 = pd.read_excel(comment_file,sheet_name='Survey')
                df2 = pd.read_excel(comment_file,sheet_name='Comments')
            else:
                exception_error = "Please upload a valid file format .xls, .xlsx or .csv"
                return render(request, "index.html", context = {'exception_error': exception_error})
            
            df1.dropna(inplace=True)
            df2.dropna(inplace=True)
            df1.reset_index(inplace=True)
            df2.reset_index(inplace=True)
    
            
            #df1 = df1.loc[df1['Contact type'] == 'Chatbot',:]
            #df1.reset_index(inplace=True)
            #df1.drop('index',inplace = True,axis = 1)
            
    
            df1['Comments'] = np.nan
    
            for i in range(len(df1)):
                for j in range(len(df2)):
                    if df1.loc[i,'Ticket'] == df2.loc[j,'Ticket']:
                        df1.loc[i,'Comments'] = df2.loc[j,'Comments']
                        break
            print("Printing df1 with comments: ", df1)
            df1.dropna(subset = ['Comments'],inplace=True)
            print(df1)

            translate_class.df = df1
            
            configParser = configparser.RawConfigParser()   
            configFilePath = r'./translate/translation.txt'
            
            configParser.read(configFilePath)
            subscription_key = configParser.get('translations', 'subscription_key')
            subscription_region = configParser.get('translations', 'subscription_region')
            base_url = configParser.get('translations', 'base_url')
            path = configParser.get('translations', 'path')
            
            
       
            translate_class.df['Translated_Comments'] = np.nan
            translate_class.df['Survey Date'] = translate_class.df['Survey Date'].astype(str)
    
            translation_language = 'en'
            
            params = '&to=' + translation_language
            constructed_url = base_url + path + params
    
            headers = {
                'Ocp-Apim-Subscription-Key': subscription_key,
                'Ocp-Apim-Subscription-Region': subscription_region,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }
            print(translate_class.df)
        
            if os.path.exists(r'..\Input_File\Feedback_History.xls'):
                df_history = pd.read_excel(r'..\Input_File\Feedback_History.xls')
            else:
                df_history = pd.DataFrame(columns=['Ticket','Survey Date','Satisfaction Level','Location','Comments','Translated_Comments','Contact Type'])
    
            count = -1
    
            if len(df_history) == 0: #If feedback history file is empty
                for i in range(len(translate_class.df)):
                    print(i)
                    lang_comment = self.detect_lang(translate_class.df.loc[i,'Comments']) #check comment language
                    if lang_comment != 'en':    # If not English, call API
                        text_input = translate_class.df.loc[i,'Comments']
                        # You can pass more than one object in body.
                        body = [{
                            'text' : text_input
                        }]
                        response = requests.post(constructed_url, headers=headers, json=body)
                        response1 = response.json()
                        print(response1)
                        translated_text = response1[0]['translations'][0]['text']
                        translate_class.df.loc[i,'Translated_Comments'] = translated_text
                    else: # If English, Use the same
                        translate_class.df.loc[i,'Translated_Comments'] = translate_class.df.loc[i,'Comments']
                df_temp = translate_class.df[['Ticket','Survey Date','Satisfaction Level','Location','Comments','Translated_Comments','Contact Type']]
                df_history = df_history.append(df_temp, ignore_index=True)
            else:   #If history file has entries
                for i in range(len(translate_class.df)):
                    
                    lang_comment = self.detect_lang(translate_class.df.loc[i,'Comments']) #check comment language
                    if lang_comment != 'en':  # If not English, check in history file
                        for j in range(len(df_history)):
                            if translate_class.df.loc[i,'Ticket'] == df_history.loc[j,'Ticket']:
                                print(i, j , "Entered")
                                translate_class.df.loc[i,'Translated_Comments'] = df_history.loc[j,'Translated_Comments']
                                count = 1
                                break
                            else:
                                count = -1
                        if count <0:
                            print(i, j, "API Call")
                            text_input = translate_class.df.loc[i,'Comments']
                            # You can pass more than one object in body.
                            body = [{
                                'text' : text_input
                            }]
                            response = requests.post(constructed_url, headers=headers, json=body)
                            response1 = response.json()
                            translated_text = response1[0]['translations'][0]['text']
                            translate_class.df.loc[i,'Translated_Comments'] = translated_text
                    else:
                        translate_class.df.loc[i,'Translated_Comments'] = translate_class.df.loc[i,'Comments']
                df_temp = translate_class.df[['Ticket','Survey Date','Satisfaction Level','Location','Comments','Translated_Comments','Contact Type']]
                df_history = df_history.append(df_temp, ignore_index=True)
            df_history.drop_duplicates(subset=['Ticket'], inplace=True)
            df_history.to_excel(r'..\Input_File\Feedback_History.xls', index=False)
    
            df_output = translate_class.df[['Ticket','Survey Date','Satisfaction Level','Location','Comments','Translated_Comments']]
            
            df_comm = translate_class.df[['Translated_Comments']]
            print(df_comm)
            category,sentiment = self.obj_classify_comm.classify(df_comm)
    
            df_output['Category'] = category
            translate_class.df['Category'] = category
            print('category')
            print(df_output)
            
    
            df_output['Sentiments'] = sentiment
            translate_class.df['Sentiments'] = sentiment
    
            df_output = df_output[['Ticket','Survey Date','Satisfaction Level','Sentiments','Location','Comments','Translated_Comments','Category']]
            df_output.to_excel(r'..\Input_File\Feedback.xls', index=False)
            content = {}
            print("df_output: ", df_output)
            
            for i in range(len(df_output)):
                content.update( {i : df_output.iloc[i].values.tolist()} )
            
        return render(request,
                      "selection.html")
    
    def filter_results(self,request):
        if request.method == "POST":
            contact_type = request.POST.get("channel_input")
            print("Filtering results: ",contact_type)
            filtered_df = translate_class.df[(translate_class.df['Contact Type'] == contact_type)]
            

            df_output = filtered_df[['Ticket','Survey Date','Satisfaction Level','Sentiments','Location','Comments','Translated_Comments','Category']]

            content = {}

            for i in range(len(df_output)):
                content.update( {i : df_output.iloc[i].values.tolist()} )

            return render(request,
                      "result_1.html", 
                      {'reservations': content})
    
    def download(self,request):
    
        excel_file = BytesIO()
        xlwriter = pd.ExcelWriter(excel_file, 'xlsxwriter')
        translate_class.df.to_excel(xlwriter, 'Feedback Analysis')
    
        xlwriter.save()
        xlwriter.close()
    
        excel_file.seek(0)
        response = HttpResponse(excel_file.read(),content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename = Feedback_Analysis.xlsx'
    
        return response

def main_classification():
    obj_classification = classification_class()
    return obj_classification

def main_translate():
    obj_translate = translate_class()
    return obj_translate