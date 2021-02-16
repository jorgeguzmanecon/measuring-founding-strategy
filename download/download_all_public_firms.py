import traceback
import re
import time
import pandas as pd
import pdb
import sys
import os
import argparse
sys.path.append(os.path.abspath('../crawler'))
from waybackmachine_crawler import waybackmachine_crawler
import requests
from data_reader import data_reader

public_output_folder = "/shared/share_scp/startup_strategy/out_public"



 
    
def read_already_downloaded():
    try:
        dirs = os.listdir(public_output_folder)
    except OSError:
        return []
    return dirs



def clean_website(website):
    return re.sub("www\.","",website)


def store_last_company(counter):
    global counter_file_path
    f = open(counter_file_path,"w")
    f.write("{0}".format(counter))


def get_last_company():
    global counter_file_path

    if counter_file_path is None or not os.path.exists(counter_file_path):
        counter_file_path = "temp.zero_counter.txt"
        store_last_company(0)
        return 0
    
    f = open( counter_file_path)
    x = f.readlines()[0]
    f.close()
    return int(x)


def download_all(websites):
    global company_index_track
    global public_output_folder
    
    counter = 0
    total_websites = websites.shape[0]
    
    print("\n\n\nStarting the scraping of all websites.  A total of {0} websites\n\n".format(total_websites))
    downloaded = read_already_downloaded()

    print("{0} already downloaded.".format(len(downloaded)))
    last_company = get_last_company()
    
    for index, company in websites.iterrows():    
        counter += 1
        if counter < company_index_track or counter < last_company:
            continue
        
        
        print("\nStarting crawl number {0} of {1} : {2}".format(counter,total_websites, company['website']))
        if clean_website(company['website']) in downloaded:
            print(".Skipping {0}. Already downloaded".format(company['website']))
            continue

        
        crawler = waybackmachine_crawler(company['website'], output_folder=public_output_folder , year_folder = True)
        ipoyear = int(company['ipoyear'])

        
        for year in range(max(ipoyear,2003),2021):
            print("\t .Starting Year: {0}".format(year))
            crawler.crawl_from_date(year,12,31)

        company_index_track = counter
        store_last_company(counter)
          

##Main
company_index_track = 0
count_errors = 0

parser = argparse.ArgumentParser()
parser.add_argument("--counter_file")
parser.add_argument("--website")
parser.add_argument("--year_missing")
parser.add_argument("--shuffle", action = "store_true")
parser.add_argument("--ignore_tracker", action = "store_true")
cmdargs = parser.parse_args()

counter_file_path = "public_current_company.txt"
if  cmdargs.counter_file is not None:
    counter_file_path = cmdargs.counter_file

if cmdargs.ignore_tracker is None:
    counter_file_path = None


if cmdargs.website is not None:
    websites = data_reader.read_public_companies(cmdargs.website)
elif cmdargs.year_missing is not None:
    websites = data_reader.find_missing_public(cmdargs.year_missing)
else:
    websites = data_reader.read_public_companies()
    

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
    

    except  (requests.exceptions.ConnectionError, requests.exceptions.MissingSchema) as e:
        count_errors += 1
        print("Exception thrown for {0} occasion. \n\tError: {1}".format(count_errors, e))
        traceback.print_exc()
        print("\n\n.. Sleeping for 1 mins..")
        
        time.sleep(60)
    

        
    except (requests.exceptions.InvalidURL) as e:
        continue
    
    else:
        break
    
