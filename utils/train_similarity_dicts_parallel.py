import sys
import pdb
import os
sys.path.append(os.path.abspath('../download'))
sys.path.append(os.path.abspath('../text_analysis'))
from data_reader import data_reader
import argparse
from similarity_estimator import similarity_estimator
from HP_industries_estimator import HP_industries_estimator

#
#  This is a utility created to train all the website novelty scores 
#
#




def train_website_novelty_scores(restimate_only, train, year):

    print("\n\n"+"*"*50)
    print("***  \tStarting year {0} ".format(year), flush=True)
    print("*"*50+ "\n\n")
    
    estimator = similarity_estimator()
    
    if not restimate_only:
        #Then, reload from underlying data
        cb_startups = data_reader.read_crunchbase()
        cb_startups['incyear'] = cb_startups.founding_year
        cb_startups = cb_startups[cb_startups.incyear == year]
        cb_startups['year'] = None
        cb_startups['path'] = "../../out/"
        cb_startups = cb_startups[['website','year','path','incyear']]
        cb_startups["type"] = "startup"
        cb_startups['source'] = "crunchbase"
        
        public_firms = data_reader.read_public_companies()
        public_firms = public_firms[public_firms.ipoyear <= year]
        public_firms['year'] = year
        public_firms['path'] = "../../out_public/"
        public_firms = public_firms[['website','year','path']]
        public_firms['type'] = "public_firm"
        public_firms['source'] = "orbis"
        
        all_websites = cb_startups.append(public_firms)
        
        estimator.load_train(all_websites)
        estimator.prepare_train_documents()

    else:
        print("No new estimates, loading old models")
        estimator.load_model("../../tfidf/{0}".format(year))

    if train:
        estimator.train()
            
    estimator.estimate_similarities()
    estimator.store_model("../../tfidf/{0}".format(year))



############## Main Code #########################


# Set to False to do the whole process again


if len(sys.argv) > 1 and sys.argv[1] != "undefined":
    year = int(sys.argv[1])
else:
    year = 2003


    
print("starting at year {0}".format(year), flush=True)

train_website_novelty_scores(restimate_only = False, train= True, year = year)
