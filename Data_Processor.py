from sklearn.model_selection import train_test_split

class data_processor:
    def __init__(self, tokenized_df, vocab, test_size, train_size, threshold, most):
        self.vocab = vocab
        self.most = most
        self.threshold = threshold
        normalized = tokenized_df['tweet'].apply(data_processor.sentence_normalizer, args=(self,))
        normalized_df = tokenized_df.assign(tweet = normalized)
        numerized = normalized_df['tweet'].apply(data_processor.numerizer, args=(self,))
        numerized_df = normalized_df.assign(numerized_tweet = numerized)
        train_valid, self.test = train_test_split(numerized_df, test_size=test_size)
        self.train, self.validate = train_test_split(train_valid, train_size=train_size)


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
                if token.norm in self.vocab.tracker:
                    if self.vocab.tracker[token.norm]>= self.threshold:
                        base.append(self.vocab.word_idx[str(token)])
                    else:
                        base.append(self.vocab.word_idx['<UNK>'])
                else:
                    base.append(self.vocab.word_idx['<UNK>'])
            except:
                base.append(self.vocab.word_idx['<PAD>'])

        return base