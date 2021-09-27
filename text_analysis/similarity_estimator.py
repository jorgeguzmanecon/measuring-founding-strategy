import traceback
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
from sklearn.metrics.pairwise import cosine_similarity
from website_text_dataset import website_text_dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from langdetect import detect


class similarity_estimator:

    def __init__(self, website_df = None):

        if website_df is not None:
            self.load_train(website_df)
        pass
        

    def load_train(self, website_df):
        (website_info, websites) = website_text_dataset.setup_website_text_df(website_df)
        self.website_info = website_info
        self.websites = websites
        

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






    

    def estimate_w2v_cosine_similarity(self):
        #this section follows this guide: https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630

        print("\t.Estimating Word2Vec Similarity", flush=True)
        
        tokenized_len = 1000
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.train_documents)
        tokenized_documents=tokenizer.texts_to_sequences(self.train_documents)
        tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=tokenized_len,padding='post')
        vocab_size=len(tokenizer.word_index)+1


        tfidfvectoriser=TfidfVectorizer()
        tfidfvectoriser.fit(self.train_documents)
        tfidf_vectors=tfidfvectoriser.transform(self.train_documents)
        words = tfidfvectoriser.get_feature_names()

        
        embedding_vector_len = self.word2vec_model.wv.vectors.shape[1]

        # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
        embedding_matrix=np.zeros((vocab_size,embedding_vector_len))
        for word,i in tokenizer.word_index.items():
            if word in self.word2vec_model.wv:
                embedding_matrix[i]=self.word2vec_model.wv[word]


        
        total_docs = len(self.documents)
        #mat = self.word2vec_model.wv.get_normed_vectors()
        #w2v_sim = cosine_similarity(mat,mat)
        document_word_embeddings = np.zeros( (total_docs, tokenized_len, embedding_vector_len))
        for i in range(total_docs):            
            for j in range(tokenized_len):
                try:
                    token_ij = tokenized_paded_documents[i,j]
                    if token_ij > 0:
                        word = tokenizer.index_word[token_ij]
                        document_word_embeddings[i,j]+=embedding_matrix[token_ij]*tfidf_vectors[i,j]
                except KeyError as e:
                    traceback.print_exc()
                    print(i)
                    print(j)
                    pdb.set_trace()

        print("Matrix of embeddings created at word level")
                    
        w2v_sim =cosine_similarity(document_word_embeddings[0])
        return w2v_sim_
        


    
    def estimate_doc2vec_similarity(self):
        mat = self.word2vec_model.docvecs.get_normed_vectors()
        w2v_sim = np.dot(mat, mat.T)

        return w2v_sim
 
    
    def estimate_allbutthetop_similarity(self):
        #Includes the all but the top post-processing steps.
        #https://openreview.net/pdf?id=HkuGJ3kCb
        orig_mat = self.word2vec_model.docvecs.get_normed_vectors()
        mat = np.subtract(orig_mat,np.mean(orig_mat,axis=0))
        abt_sim = cosine_similarity(mat)
        return abt_sim


    def estimate_doc2vec_euclidean_dist(self):

        mat = self.word2vec_model.docvecs.get_normed_vectors()
        ecl_sim = euclidean_distances(mat, mat)
        return ecl_sim



    


    
    def get_row_similarities(self, sim, prefix, i):
        try:
            pdrow = {}
             
            ### Similarity to Startups
            startups_index = website_text_dataset.get_valid_startups_index(self.website_info)
            self_index= website_text_dataset.get_self_index(self.website_info, i)
            matches = np.all([startups_index,~self_index], axis=0)
            
            srow = sim[i][np.where(matches)]
            srow = np.sort(srow)        
            pdrow[prefix+'sim_startup_1'] = srow[-1]
            pdrow[prefix+'sim_startup_3'] = np.average(srow[-3:])
            pdrow[prefix+'sim_startup_5'] = np.average(srow[-5:])
            
            pdrow[prefix+'sim_startup_median'] = np.median(srow)
            pdrow[prefix+'hp_startup_firm_5'] =  np.average(srow[-5:]) - pdrow[prefix+'sim_startup_median']


            ### Similarity to public firms
            public_firms_index = website_text_dataset.get_valid_public_firms_index(self.website_info) 
            self_index= website_text_dataset.get_self_index(self.website_info, i, firmtype="public_firm")
            matches = np.all([public_firms_index, ~self_index], axis=0)

            prow = sim[i][np.where(matches)]
            prow = np.sort(prow)        
            pdrow[prefix+'sim_public_firm_1'] = prow[-1]
            pdrow[prefix+'sim_public_firm_3'] = np.average(prow[-3:])
            pdrow[prefix+'sim_public_firm_5'] = np.average(prow[-5:])

            pdrow[prefix+'sim_public_median'] = np.median(prow)
            pdrow[prefix+'hp_public_firm_5'] =  np.average(prow[-5:]) -  pdrow[prefix+'sim_public_median'] 



            
        except IndexError:
            import sys , traceback
            sys.exc_info()[0]
            traceback.print_exc()
            pdb.set_trace()
        return pdrow

    
    def create_estimate_row(self, similarities,prefixes, index, debug_data = False):
        pdrow = {}        
        pdrow['website_name'] = self.website_info.iloc[index,0]
        pdrow['website_len'] = self.website_info.iloc[index,1]

        for j in range(len(similarities)):            
            pdrow.update(self.get_row_similarities(similarities[j], prefix=prefixes[j],i=index))

        return(pdrow)
    




    # next frontier is to include the following info
    #                       https://openreview.net/pdf?id=HkuGJ3kCb
    
    def estimate_similarities(self, debug_data = False):
      
        self.website_info = website_text_dataset.prep(self.website_info)
        print("\t.Estimating linear kernel similarity")
        tfidf_sim = linear_kernel(self.tfidf_model, self.tfidf_model)
        print("\t.Done")

        w2v_sim = self.estimate_doc2vec_similarity()
        abt_sim = self.estimate_allbutthetop_similarity()
        ecl_sim = self.estimate_doc2vec_euclidean_dist()*-1 #make negative so it still all works as similarity
        rows_list = []

        startup_ids = np.where(self.website_info.type == "startup")[0]
        print("Getting similarities. Total: {0}".format(len(startup_ids)))
        for i in startup_ids:
        #        for i in range(0,sim.shape[1]):
            if (i%25) == 0:
                print("{0} ".format(i),end='', flush=True)

            pdrow = self.create_estimate_row( similarities=[tfidf_sim, w2v_sim, abt_sim, ecl_sim],
                                              prefixes = ['tfidf_','w2v_','abt_','ecl_'],
                                              index = i)
            rows_list.append(pdrow)

        self.similarity_scores = pd.DataFrame(rows_list)
        self.w2v_sim_matrix = w2v_sim
        self.abt_sim_matrix = abt_sim
        print("")




    
    def get_most_similar_firms(self,verbose=True, debug=False):
                
        self.website_info = website_text_dataset.prep(self.website_info)
        
        if(self.abt_sim_matrix is None):
            print("ERROR: get_most_similar_firms can only be run after estimate_similarities")
            return (None)
        public_firms_index = website_text_dataset.get_valid_public_firms_index(self.website_info)       
        startups_index = website_text_dataset.get_valid_startups_index(self.website_info)

        similar_firms = pd.DataFrame()
        print("Adding similar firms into a dataset")
         
        for i in np.where(startups_index)[0]:
            if (i % 25) == 0:
                print("{0}/{1} ".format(i, similar_firms.shape[0]),end='', flush=True)

            ### Top Public Firms Matched
            self_index= np.all([self.website_info.website == self.website_info.website[i],
                                self.website_info.type == "public_firm"],
                               axis=0)
            matches = np.all([public_firms_index, ~self_index], axis=0)
            public_firms = self.website_info.loc[matches].copy()
            public_firms['sim'] =self.abt_sim_matrix[i][matches]

            focal_website = self.website_info.iloc[i]
            public_firms['focal_website'] = focal_website.website
            public_firms['focal_website_text'] = focal_website.text
            public_firms['focal_website_snapshot_in_window'] = focal_website.snapshot_in_window
            
            public_firms.sort_values(by='sim',ascending=False, inplace=True)
            similar_firms = similar_firms.append(public_firms.head(5))

            if verbose:
                print("\n\n ***************** {0} *********************\n\n".format(self.website_info.website[i]))
                print("{0}  {1}".format(focal_website.website, focal_website.text[0:255]))
                print("\n\n")
                print(public_firms[['sim','text','website']].head(5))
                
            
            ### Top Startups Matched
            self_index= np.all([self.website_info.website == self.website_info.website[i],
                               self.website_info.type == "startup"],
                               axis=0)            
            matches = np.all([startups_index,~self_index], axis=0)
            startups = self.website_info.loc[matches].copy()
            startups['sim'] =self.abt_sim_matrix[i][matches]

            startups['focal_website'] = focal_website.website
            startups['focal_website_text'] = focal_website.text
            startups['focal_website_snapshot_in_window'] = focal_website.snapshot_in_window           
            startups.sort_values(by='sim',ascending=False, inplace=True)
            
            similar_firms = similar_firms.append(startups.head(5))

            if verbose:
                print(startups[['sim','text','website']].head(5))

            
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

        print("\t. Estimating Word2Vec model", flush=True)
        
        print("\t. Loading training documents", flush=True)

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
        model = Doc2Vec(documents = documents,
                        vector_size=700,
                        window=7,
                        min_count =3)
        print("\t. Done")
        self.word2vec_model = model
        self.documents = all_docs

        vec = TfidfVectorizer()
        model = vec.fit(self.train_documents)
        self.idf_weights = dict(zip(model.get_feature_names(), model.idf_))
        

            



        

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
        
