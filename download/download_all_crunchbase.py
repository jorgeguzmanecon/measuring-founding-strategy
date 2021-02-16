import argparse
import traceback
import re
import time
import pandas as pd
import pdb
import sys
import os
sys.path.append(os.path.abspath('../crawler'))
from waybackmachine_crawler import waybackmachine_crawler
import requests
from data_reader import data_reader


def read_already_downloaded():
    try:
        dirs = os.listdir("../../out")
    except OSError:
        return []
    return dirs




def store_last_company(counter):
    f = open("./crunchbase_current_company.txt","w")
    f.write("{0}".format(counter))

def get_last_company():
    try:
        f = open("./crunchbase_current_company.txt")
        x = f.readlines()[0]
        f.close()
        return int(x)
    except FileNotFoundError:
        return 0



def download_all(websites):
    global company_index_track
    
    counter = 0
    count_downloaded = 0
    count_skipped = 0
    total_websites = websites.shape[0]
    
    print("\n\n\nStarting the scraping of all websites.  A total of {0} websites\n\n".format(total_websites))


    last_company = get_last_company()    
    for index, company in websites.iterrows():    

        if (counter % 100) == 0: 
            downloaded = read_already_downloaded()
            
        counter += 1
        if counter < company_index_track or counter < last_company:
            continue
        

        print("\nStarting crawl number {0} of {1} : {2}".format(counter,total_websites, company['website']))

        if data_reader.clean_domain_url(company['website']) in downloaded:
            print(".Skipping {0}. Already downloaded".format(company['website']))
            count_skipped += 1
            continue

        crawler = waybackmachine_crawler(company['website'])
        year = company['founding_year'] + 1
        crawler.crawl_from_date(year,1,1)

        company_index_track = counter
        store_last_company(counter)
        count_downloaded += 1
        tot = count_downloaded + count_skipped
        print("\t. -- Download done.\n\ \t. STATS: {0} Downloaded ({1}%). {2} Skipped ({3}%)".format(count_downloaded,  round(count_downloaded*100/tot), count_skipped, round(count_skipped *100/tot)))

          

##Main
company_index_track = 0
count_errors = 0
websites = data_reader.read_crunchbase()


parser = argparse.ArgumentParser()
parser.add_argument("--shuffle", action = "store_true")
cmdargs = parser.parse_args()

if cmdargs.shuffle is not None:
    print("\nShuffling website list\n")
    websites = websites.sample(frac = 1)

    
while True:
    try:
        download_all(websites)

    except (requests.exceptions.TooManyRedirects) as e:
        count_errors += 1
        print("Exception thrown for too many redirects. Skipping. \n\tError: {1}".format(count_errors, e))
        traceback.print_exc()
        print("\n\n.. Sleeping for 10 secs..")
        company_index_track += 1
        time.sleep(10)
    

    except  (requests.exceptions.ConnectionError) as e:
        count_errors += 1
        print("Exception thrown for {0} occasion. \n\tError: {1}".format(count_errors, e))
        traceback.print_exc()
        print("\n\n.. Sleeping for 10 mins..")
        
        time.sleep(10*60)
    

        
    except (requests.exceptions.InvalidURL) as e:
        continue
    
    else:
        break
    
