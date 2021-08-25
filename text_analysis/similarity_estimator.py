
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from sklearn.metrics.pairwise import linear_kernel
import nltk
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from website_text import website_text
import pdb
import pickle
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec 
from gensim.models.doc2vec import Doc2Vec , TaggedDocument

class similarity_estimator:

    def __init__(self, website_df = None):

        if website_df is not None:
            self.load_train(website_df)
        pass
        


    def load_train(self, website_df):
        websites = []
        
        website_list= []

        print("Loading all websites. Total: {0}".format(website_df.shape[0]))
        counter  = 0
        counter_good = 0
        for index , row in website_df.iterrows():
            counter += 1
            doc = website_text(row['path'], row['website'] , row['year'], row['incyear'])

            if doc is not None and doc.is_valid_website():
                counter_good += 1
                websites.append(doc)

                website_info = {}
                website_info['website'] = row['website']
                website_info['text_len'] = len(doc.get_website_text())
                website_info['source'] = row['source']

                text = doc.get_website_text()
                text = text[1:5000] if len(text) > 5000 else text
                website_info['text'] = text
            
                website_info['type'] = row['type']
                website_list.append(website_info)

            if (counter % 10) == 0:
                print("\t.. {0} ({1})".format(counter, counter_good))

        self.website_info = pd.DataFrame(website_list)    
        self.websites = websites
        print("\t Done")
        
        

    def prepare_train_documents(self):
        documents = []
        
        for w in self.websites:
            text = w.get_website_text()
            text = text[:400000] if len(text) > 400000 else text
            documents.append(text)

        self.train_documents = documents



    def get_similarity_matrix(self, new_documents):
        new_docsx = self.tfidf_model.transform(new_documents)
        sim = linear_kernel(new_docsx, new_docx)
        return sim




    def get_row_similarities(self, sim, row, i, prefix = ""):

        try:
            pdrow = {}
            pdrow[prefix+'sim_1'] = row[-1]
            pdrow[prefix+'sim_3'] = np.average(row[-3:])
            pdrow[prefix+'sim_5'] = np.average(row[-5:])
            

            startups_index = np.all(self.website_info.type == "startup",
                                    self.website_info.snapshot_in_window == True,
                                    axis=0)

            self_index= np.all(self.website_info.website == self.website_info.website[i],
                               self.website_info.type == "startup",
                               axis=0)
            
            matches = np.all(startups_index,~self_index, axis=0)
            
            srow = sim[i][matches]
            srow = np.sort(srow)        
            srow = srow[~(srow > .95)]
            pdrow[prefix+'sim_startup_1'] = srow[-1]
            pdrow[prefix+'sim_startup_3'] = np.average(srow[-3:])
            pdrow[prefix+'sim_startup_5'] = np.average(srow[-5:])
            
            pdrow[prefix+'sim_startup_median'] = np.median(srow)
            pdrow[prefix+'hp_startup_firm_5'] =  np.average(srow[-5:]) - pdrow[prefix+'sim_startup_median']
            pdrow[prefix+'hp_num_startups_in_industry'] =  np.sum(srow > .23)
        

            
            public_firms_index = self.website_info.type == "public_firm"            
            self_index= np.all(self.website_info.website == self.website_info.website[i],
                               self.website_info.type == "public_firm", ##not common, but in case it's an IPO in the same year
                               axis=0)
            matches = np.all(public_firms_index, ~self_index, axis=0)

            prow = sim[i][matches]
            prow = np.sort(prow)        
            pdrow[prefix+'sim_public_firm_1'] = prow[-1]
            pdrow[prefix+'sim_public_firm_3'] = np.average(prow[-3:])
            pdrow[prefix+'sim_public_firm_5'] = np.average(prow[-5:])

            pdrow[prefix+'sim_public_median'] = np.median(prow)
            pdrow[prefix+'hp_public_firm_5'] =  np.average(prow[-5:]) -  pdrow[prefix+'sim_public_median'] 
            
            pdrow[prefix+'hp_num_public_in_industry'] =  np.sum(prow > .23)

        except IndexError:
            import sys , traceback
            sys.exc_info()[0]
            traceback.print_exc()
            pdb.set_trace()
        return pdrow



    def create_estimate_row(self, sim, w2v_sim, i, debug_data = False):
        row = sim[i]
        row = np.sort(row)
        #Keep only similarities below 0.95
        row = row[~(row > .95)]
            
        pdrow = {}
        
        pdrow['website_name'] = self.website_info.iloc[i,0]
        pdrow['website_len'] = self.website_info.iloc[i,1]
        pdrow.update(self.get_row_similarities(sim,row,i))


        ######### Word2Vec piece
        row = w2v_sim[i]
        row = np.sort(row)
        row = row[~(row > .95)]
        pdrow.update(self.get_row_similarities(w2v_sim, row,i, prefix="w2v_"))


        return(pdrow)
    

            

    def estimate_similarities(self, debug_data = False):
        print("\t.Estimating linear kernel similarity")
        sim = linear_kernel(self.tfidf_model, self.tfidf_model)
        print("\t.Done")

        print("\t.Initializing Word2Vec Matrices")
        self.word2vec_model.init_sims()
        print("\t.Estimating Word2Vec Similarity")
        mat = self.word2vec_model.docvecs.doctag_syn0norm
        w2v_sim = np.dot(mat, mat.T)
        print("\t.Done")
        rows_list = []

        print("Getting similarities")
        for i in np.where(self.website_info.type == "startup")[0]:
        #        for i in range(0,sim.shape[1]):
            if (i%25) == 0:
                print("{0} ".format(i),end='', flush=True)

            pdrow = self.create_estimate_row( sim, w2v_sim, i)
            rows_list.append(pdrow)

        self.similarity_scores = pd.DataFrame(rows_list)
        self.w2v_sim_matrix = w2v_sim
        print("")



    
    def get_most_similar_firms(self):

        if(self.w2v_sim_matrix is None):
            print("ERROR: get_most_similar_firms can only be run after estimate_similarities")
            return (None)

        public_firms_index = self.website_info.type == "public_firm"            
        startups_index = np.all(self.website_info.type == "startup",
                                self.website_info.snapshot_in_window == True,
                                axis=0)

        similar_firms = pd.DataFrame()


        print("Adding similar firms into a dataset")
        
        for i in startups_index:
            if (i % 25) == 0:
                print("{0}/{1} ".format(i, similar_firms.shape[0]),end='', flush=True)


            ### Top Public Firms Matched
            self_index= np.all(self.website_info.website == self.website_info.website[i],
                               self.website_info.type == "public_firm",
                               axis=0)
            
            matches = np.all(public_firms_index, ~self_index, axis=0)

            public_firms = self.website_info.loc[matches].copy()
            public_firms['sim'] =self.w2v_sim_matrix[i][matches]
            public_firms = public_firms.loc[public_firms.sim < .95]
            public_firms['matched_website'] = self.website_info.website[i]

            public_firms.sort_values(by='sim',ascending=False, inplace=True)
            similar_firms = similar_firms.append(public_firms.head(5))

            
            ### Top Startups Matched
            self_index= np.all(self.website_info.website == self.website_info.website[i],
                               self.website_info.type == "startup",
                               axis=0)
            
            matches = np.all(startups_index,~self_index)
            startups = self.website_info.loc[matches].copy()
            startups['sim'] =self.w2v_sim_matrix[i][matches]
            startups = startups.loc[startups.sim < .95]
            startups['matched_website'] = self.website_info.website[i]
            startups.sort_values(by='sim',ascending=False, inplace=True)
            
            similar_firms = similar_firms.append(startups.head(5))

            
        #pdb.set_trace()
        return(similar_firms)


        
    def load_model(self,path):
        print("\t.Loading from {0}".format(path))
        self.tfidf_model = pickle.load(open(path + ".model.pickle","rb"))
        self.website_info = pd.read_pickle(path + ".website_info.pickle")
        self.similarity_scores = pickle.load(open(path + ".similarity_scores.pickle","rb"))
        self.word2vec_model = Word2Vec.load(path + ".word2vec.model")
        print("\t.Model Loaded")



    def store_model(self, path):
        print ("Storing tf idf model to {0}".format(path))
        pickle.dump(self.tfidf_model , open(path + ".model.pickle","wb"))
        self.website_info.to_pickle(path + ".website_info.pickle")
        pickle.dump(self.similarity_scores , open(path + ".similarity_scores.pickle","wb"))
        self.word2vec_model.save(path + ".word2vec.model")


    def train(self):
        self.train_tfidf()
        self.train_word2vec()


    def train_word2vec(self):
        if self.train_documents is None:
            self.prepare_train_documents()

        print("\t. Estimating Word2Vec model")            
        
        print("\t. Loading training documents")

        counter = 0
        all_docs = []
        for train_doc in self.train_documents:

            doc = train_doc[:150000] if len(train_doc) > 150000 else train_doc
            if (counter%100) == 0:
                print("{0} .. len: {1}".format(counter,len(doc)))

            counter += 1
            doc = remove_stopwords(doc)
#            doc = re.sub(r'[^\w\s]','',doc)
            doc_tokens =nltk.word_tokenize(doc.lower())
            all_docs.append(doc_tokens)            

        print("Creating all tagged documents")
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_docs)]    

        print("\t. Run model")
        model = Doc2Vec(documents = documents)
        print("\t. Done")
        self.word2vec_model = model
    



        

    def train_tfidf(self):
        if self.train_documents is None:
            self.prepare_train_documents()


        print("\t. Estimating similarity through tf-idf")
        print("\t. Creating TfidfVectorizer")
        #Update: many more ngrams allows
        vectorizer = TfidfVectorizer( decode_error = 'ignore',
                                      strip_accents = 'unicode',
                                      lowercase = True,
                                      stop_words = text.ENGLISH_STOP_WORDS,
                                      ngram_range = (1,3),
                                      max_df = .5,
                                      min_df = .005)
        print("\t. fit_transform")
        
        tfidf = vectorizer.fit_transform(self.train_documents)
        print("\t. Done")
        self.tfidf_model = tfidf
        
