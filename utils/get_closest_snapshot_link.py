import argparse
import traceback
import re
import time
import pandas as pd
import pdb
import sys
import os
sys.path.append(os.path.abspath('../crawler'))
sys.path.append(os.path.abspath('../text_analysis'))
from waybackmachine_crawler import waybackmachine_crawler
import requests
from data_reader import data_reader

from json.decoder import JSONDecodeError
 

###############################################
##
##
##   This script goes to the Wayback Machine to get the closest website link
##    for all websitse in Crunchbase.
##
##
##
##
##
##


websites = data_reader.read_crunchbase()
websites[['closest_snapshot']] = ""
websites[['closest_snapshot_time']] = ""


for index, company in websites.iterrows():
    crawler = waybackmachine_crawler(company['website'])
    year = company['founding_year'] + 1

    try:
        closest_snapshot = crawler.list_closest_snapshot(year,1,1)
    except JSONDecodeError:
        print("\n\n*********JSONDecodeError************")

    if closest_snapshot is not None:
        websites.at[index,'closest_snapshot']=str(closest_snapshot)
        websites.at[index,'closest_snapshot_time']=closest_snapshot['timestamp']
    else:
        websites.at[index,'closest_snapshot']="None"
    
    print("\t\t---->  {0} of {1}".format(index,websites.shape[0]))

    #store every 100 sites
    if (index%100) == 0:
        websites.to_stata("../../tfidf/closest_snapshots_list.dta", version = 117)
    
websites.to_stata("../../tfidf/closest_snapshots_list.dta", version = 117)


