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

for year in range(2003,2018):
    print("\n\nDoing year {0}".format(year))
    estimator = similarity_estimator()
    estimator.load_model("../../tfidf/{0}".format(year))
    estimator.estimate_similarities()
    #pdb.set_trace()
    similar_websites = estimator.get_most_similar_firms()

    path = "../../revision/most_similar_websites_{0}.dta".format(year)
    similar_websites['text'] = similar_websites['text'].str.encode('latin-1','ignore')
    similar_websites['text'] = similar_websites['text'].astype(str)
    similar_websites.snapshot_in_window.replace(np.nan,0,inplace = True)
    similar_websites['snapshot_in_window'] = similar_websites.snapshot_in_window.astype(int)
    similar_websites.to_stata(path, version = 118)

