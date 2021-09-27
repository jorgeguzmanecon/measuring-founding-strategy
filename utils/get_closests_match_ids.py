import sys
import pdb
import os
sys.path.append(os.path.abspath('../download'))
sys.path.append(os.path.abspath('../text_analysis'))
from data_reader import data_reader

from similarity_estimator import similarity_estimator
from HP_industries_estimator import HP_industries_estimator
import numpy as np
import pandas as pd

# environment settings: 
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 80)
pd.set_option('expand_frame_repr', True)
pd.set_option('display.width', 170)

similar_websites_all = None

for year in range(2003,2019):
    print("\n\nDoing year {0}".format(year))
    estimator = similarity_estimator()
    estimator.load_model("../../tfidf/{0}".format(year))
    estimator.estimate_similarities()
    
    similar_websites = estimator.get_most_similar_firms(verbose=False)


    similar_websites['text'] = similar_websites['text'].str.encode('latin-1','ignore')
    similar_websites['text'] = similar_websites['text'].astype(str)

    similar_websites = similar_websites.loc[similar_websites.focal_website_snapshot_in_window]      

    similar_websites['year'] = year
    similar_websites['focal_website_snapshot_in_window'] = similar_websites.focal_website_snapshot_in_window.astype(float)
    similar_websites['snapshot_in_window'] = similar_websites.snapshot_in_window.astype(float)
    
    if similar_websites_all is None:
        similar_websites_all = similar_websites
    else:
        similar_websites_all = similar_websites_all.append(similar_websites)

        
    path = "../../data_output/most_similar_websites.dta"
    similar_websites_all.to_stata(path, version = 118)
    print("Similar websites stored to {0}".format(path))

