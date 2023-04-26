# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """
    
    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess):
        raise NotImplementedError("Subclasses of Feature must implement this function")

class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    
    def __call__(self, question, run, guess):
        # How many characters long is the question?
        yield ("char", log(1 + len(run)))
        
        # How many words long is the question?
        yield ("word", log(1 + len(run.split())))

        # How many characters long is the guess?
        yield ("guess", log(1 + len(guess)))


class FrequencyFeature(Feature):                       

    def __init__(self, name):                 
        from buzzer import normalize_answer   
        self.name = name                      
        self.counts = Counter()               
        self.normalize = normalize_answer


    def in_data(self, guess):    
        import json

        with open("../data/wiki_page_text.json") as infile: 

            guesses = json.load(infile)
            new_guess = guess.replace(" ", "_")               

            return (new_guess in guesses)

    def __call__(self, question, run, guess):                               
	    yield ("guess", self.counts[guess])


class FinalPOS(Feature):

    """
    Feature that computes the POS of the final word in run which is the word that
    was heard before the buzz in occurred.
    """

    def __init__(self, name):
        self.name = name

    def get_finalpos(self, text):
        from nltk.tokenize import word_tokenize
        import nltk
        nltk.download('averaged_perceptron_tagger')
        new_text = word_tokenize(text)
        lst = nltk.pos_tag(new_text)
        length = len(lst)

        return (lst[length - 1])[1]

    def __call__(self, question, run, guess):
        yield ("POS", self.get_finalpos(run))


class Expected_Length(Feature):

    """
    Feature that looks at the length of the guess and sees how close it is for what is expected for this type of question
    """

    def __init__(self, name):
        self.name = name
        self.dict = {}

    def add_training(self, file):
        import statistics
        import json
        from buzzer import normalize_answer

        with open(file) as infile:                    
            questions = json.load(infile)
            dict = {} 

            for ii in questions:
                word = None

                if ("this" in ii["text"].split()):
                    lst = (ii["text"].split())
                    word = normalize_answer(lst[(lst.index("this")) + 1])
                elif ("This" in ii["text"].split()):
                    lst = (ii["text"].split())
                    word = normalize_answer(lst[(lst.index("This")) + 1])

        
                if word != None:
                    if word in dict:
                        dict[word].append(len(ii["answer"].split()))
                    else:
                        dict[word] = [len(ii["answer"].split())]
    
    
            for key in dict.keys():
                dict[key] = statistics.median(dict[key])

            self.dict = dict

    
    def get_expected(self, text):
        import statistics
        from buzzer import normalize_answer
        base_value = statistics.mean(self.dict.values())

        word = None

        if ("this" in text.split()):
            lst = (text.split())
            
            if (lst.index("this") + 1) >= len(lst):
                return base_value

            word = normalize_answer(lst[(lst.index("this")) + 1])
        elif ("This" in text.split()):
            lst = (text.split())

            if (lst.index("This") + 1) >= len(lst):
                return base_value

            word = normalize_answer(lst[(lst.index("This")) + 1])

        if ((word == None) or (word not in self.dict)):
            return base_value
        else:
            return self.dict[word]
        

    def __call__(self, question, run, guess):
        yield ("Expected", abs(len(guess.split()) - self.get_expected(run)))           
	

class Category_Length(Feature):
    def __init__(self, name):
        self.name = name
        self.classifier = None

    def length(self, text):
        return {"length": len(text.split())}

    def add_training(self, file):
        import nltk
        import json

        with open(file) as infile:                   
            questions = json.load(infile)
            lst = [(self.length(ii["answer"]), ii["category"]) for ii in questions]
            

            train_set = lst
            classifier = nltk.NaiveBayesClassifier.train(train_set)
            self.classifier = classifier

    def __call__(self, question, run, guess):
        yield("Category_Match", (self.classifier.classify(self.length("guess")) == question["category"]))


class Answer_Similarity(Feature):
    def __init__(self, name):
        self.name = name
        self.tfidf = None
        self.sims = None
        self.dictionary = None
        self.lookup = {}
    
    def get_data(self):
        from nltk.tokenize import word_tokenize
        import os
        os.system("pip install gensim")
        import gensim
        import json

        with open("../data/wiki_page_text.json") as infile:
            file_docs = []
            questions = json.load(infile)
            index = 0

            for ii in questions:
                lst = ii["text"].split(".")[:30]

                sentence = ' '.join(lst)

                token = word_tokenize(sentence)
                file_docs.append(token)
                self.lookup[ii["page"]] = index
                index += 1

    
            dictionary = gensim.corpora.Dictionary(file_docs)
            corpus = [dictionary.doc2bow(gen_doc) for gen_doc in file_docs]
            tf_idf = gensim.models.TfidfModel(corpus)
    
            sims = gensim.similarities.Similarity(os.getcwd(), tf_idf[corpus],
                                        num_features=len(dictionary))

            
            self.sims = sims
            self.tfidf = tf_idf
            self.dictionary = dictionary

    def get_similarity(self, guess, run):
        from nltk.tokenize import word_tokenize

        new_guess = guess.replace(" ", "_")

        if new_guess not in self.lookup:
            return 0

        txt = run
        file_docs2 = [word_tokenize(txt)]
    
        for line in file_docs2:
            query_doc_bow = self.dictionary.doc2bow(line)

        query_doc_tf_idf = self.tfidf[query_doc_bow]
        lst = self.sims[query_doc_tf_idf].tolist()
        
        return lst[self.lookup[new_guess]]
    
    def __call__(self, question, run, guess):
        yield("Answer_Sim", self.get_similarity(guess, run))


class inQuestion(Feature):
    def __init__(self, name):
        self.name = name

    def in_prompt(self, guess, run):
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(guess)

        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

        filtered_sentence = []
  
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        count = 0

        for word in filtered_sentence:
            if word in run:
                return True
        
        return False
    
    def __call__(self, question, run, guess):
        yield("In_Prompt", self.in_prompt(guess, run))

    
class isEmpty(Feature):
    def __init__(self, name):
        self.name = name
    
    def __call__(self, question, run, guess):
        yield("Empty", (len(guess) < 2))





