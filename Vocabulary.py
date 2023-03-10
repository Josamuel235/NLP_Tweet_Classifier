class vocab_builder:
    def __init__(self, tokenized_df):
        self.longest= 0
        self.idx_word= {}
        self.word_idx = {}
        self.tracker = {}
        tokenized_df['tweet'].apply(vocab_builder.vocab_gen, args=(self,))
        self.word_idx['<PAD>'] = len(self.word_idx)
        self.idx_word[len(self.idx_word)] = '<PAD>'
        self.word_idx['<UNK>'] = len(self.word_idx)
        self.idx_word[len(self.idx_word)] = '<UNK>'
        
    def vocab_gen(sentence, self):
        count = 0 
        for word in sentence:
            count +=1
            if word.norm not in self.tracker:
                self.tracker[word.norm] = 1
                self.word_idx[str(word)] = len(self.word_idx)
                self.idx_word[len(self.idx_word)] = str(word)
            else:
                self.tracker[word.norm] += 1
        if self.longest < count:
            self.longest = count
        return 