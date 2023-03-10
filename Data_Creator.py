import pandas as pd 
import spacy

class Data_Creator:
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        data = self.clean(data)
        self.tokenized_df = self.tokenize(data)
        

    def tokenize(self, df):
        nlp = spacy.load("en_core_web_sm")
        tweet = df['tweet'].apply(lambda x: nlp(x.strip()))
        tokenized = df.assign(tweet = tweet)
        return tokenized
    
    def clean(self, data):
        
        repl = {'@\w*': ' ', '&amp;' : 'and','\su\s':' you ', '&#\w*;': ' ', 
        '#':' ', '\s2\s': 'two', 'bihday':"birthday", "ð[^ ]*": ' ' ,
        "â[^ ]*": ' ',"(dont)|(don't)": 'do not', "(cant)|(can't)": "can not",
        "(yous)|(you's)": "you is", "(yous)|(you's)": "you is", 
        "(youve)|(you've)": "you have", "(doesnt)|(doesn't)": 'does not', 
        "(wont)|(won't)": 'will not', "[0-9]+\.*[0-9%]+\w*" : "NUMBER",'\\n\.':' ' ,'\\n':' ',
        "\.{2,}": '.', "!{2,}":'!', "\?{2,}":'?', 'ing[^a-z]':' ', 'ed[^a-z]': ' ', '_':" ",
        ' +': ' '}

        cleaned_tweet = data['tweet'].str.lower()
        cleaned_tweet = cleaned_tweet.replace(repl, regex=True)
        cleaned = data.assign(tweet = cleaned_tweet)
        return cleaned.drop("Unnamed: 0", axis=1)