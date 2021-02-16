import sys
import pdb
import os
sys.path.append(os.path.abspath('../download'))
import terminal_banner
from data_reader import data_reader

from similarity_estimator import similarity_estimator



for year in range(2003, 2018):
    estimator = similarity_estimator()
    estimator.load_tfidf("../../tfidf/{0}.pickle".format(year))
    
