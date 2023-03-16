import gradio as gr
import spacy
import pandas as pd
import numpy as np
import pickle
import torch
from LSTM_Architecture import NLP_LSTM
import warnings

warnings.filterwarnings('ignore')

nlp = spacy.load("en_core_web_sm")
file = open('tok_to_work', 'rb')
tok_convert = pickle.load(file)
file.close()

file = open('tok_track', 'rb')
tok_track = pickle.load(file)
file.close()

gpu = torch.cuda.is_available()
model = NLP_LSTM(False, 150, 150, len(tok_convert), 4, 0.44, True)
info = torch.load('latest_model.pt', map_location=torch.device('cpu'))
model.load_state_dict(info['model'])

class data_processor:
    def __init__(self, tokenized_df, conv, track):
        self.word_idx = conv
        self.tracker = track
        self.most = 29
        self.threshold = 5
        normalized = tokenized_df['tweet'].apply(data_processor.sentence_normalizer, args=(self,))
        normalized_df = tokenized_df.assign(tweet = normalized)
        numerized = normalized_df['tweet'].apply(data_processor.numerizer, args=(self,))
        self.text = normalized_df.assign(numerized_tweet = numerized)


    def sentence_normalizer(sentence, self):
        final_tok = []
        count = 0 
        for token in sentence:
            final_tok.append(token)
            count+=1
            if count >= self.most:
                break
        if len(final_tok)<self.most:
            final_tok.extend(['<PAD>']*(self.most-len(final_tok)))
        return final_tok


    def numerizer(x, self):
        base = []
        for token in x:
            try:
                if token.norm in self.tracker:
                    if self.tracker[token.norm]>= self.threshold:
                        base.append(self.word_idx[str(token)])
                    else:
                        base.append(self.word_idx['<UNK>'])
                else:
                    base.append(self.word_idx['<UNK>'])
            except:
                base.append(self.word_idx['<PAD>'])

        return base

class GUI:
    def __init__(self, nlp, tok_convert, tok_track, model):
        self.nlp = nlp
        self.tok_convert = tok_convert
        self.tok_track = tok_track
        self.model = model


    def run(self, Text):
        repl = {'@\w*': ' ', '&amp;' : 'and','\su\s':' you ', '&#\w*;': ' ', 
        '#':' ', '\s2\s': 'two', 'bihday':"birthday", "ð[^ ]*": ' ' ,
        "â[^ ]*": ' ',"(dont)|(don't)": 'do not', "(cant)|(can't)": "can not",
        "(yous)|(you's)": "you is", "(yous)|(you's)": "you is", 
        "(youve)|(you've)": "you have", "(doesnt)|(doesn't)": 'does not', 
        "(wont)|(won't)": 'will not', "[0-9]+\.*[0-9%]+\w*" : "NUMBER",'\\n\.':' ' ,'\\n':' ',
        "\.{2,}": '.', "!{2,}":'!', "\?{2,}":'?', 'ing[^a-z]':' ', 'ed[^a-z]': ' ', '_':" ",
        ' +': ' '}

        text = pd.DataFrame({'tweet': [Text]})
        cleaned = text['tweet'].str.lower()
        cleaned = cleaned.replace(repl, regex=True)
        text = text.assign(tweet = cleaned)
        cleaned = text['tweet'].apply(lambda x: self.nlp(x.strip()))
        text = text.assign(tweet = cleaned)
        tok_convert
        text = data_processor(text, self.tok_convert, self.tok_track)
        text = text.text
        text = torch.tensor(np.stack(text['numerized_tweet']))
        text = torch.round(self.model.tester(text)).item()
        if text == 1:
                return 'This is Toxic'
        else:
                return 'This is not Toxic'

## INIT

ui = GUI(nlp, tok_convert, tok_track, model)

## UI 

outputs = gr.outputs.Textbox()
inputs = gr.inputs.Textbox()
app = gr.Interface(fn=ui.run, inputs=[inputs], outputs=outputs, description="This is a Sentiment Analysis Model")

app.launch(share = True)