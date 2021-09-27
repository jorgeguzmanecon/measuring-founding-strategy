

import sys
import pdb
import os
sys.path.append(os.path.abspath('../download'))
sys.path.append(os.path.abspath('../text_analysis'))
from data_reader import data_reader
from website_text_dataset import website_text_dataset
from langdetect import detect

import argparse


for year in range(2003, 2019):

    print("Reading data for year {0}".format(year))
    
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

    print("Loading text")

    (website_info, websites) = website_text_dataset.setup_website_text_df(all_websites, truncate_text_chars=None)
    #pdb.set_trace()
    dta_path = "../../data_output/website_info_{0}.dta".format(year)
    
    print("text loaded")
    print("replacing nas")
    website_info["snapshot_in_window"] = website_info.snapshot_in_window.astype(float)
    print("storing file to {0}".format(dta_path))


    try:
        website_info.to_stata(dta_path,version=118)
    except:
        print("ERROR writing file!")
        pdb.set_trace()
