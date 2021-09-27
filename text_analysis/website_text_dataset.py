
import nltk
import pandas as pd
from website_text import website_text
import pdb
import pickle
import numpy as np
import os
from langdetect import detect




class website_text_dataset:
    # This class is created for a series of helper methods to study the whole website dataset
    # such as adding flags of whether something is valid.  The method 'prep' is supposed to be called
    # before any analysis is done, thus allowing all sorts of things to be added in the process.


    
    def prep(website_info, update_language = False):
        website_info['text'] = website_info.text.str.strip()

        
        if 'lang' not in website_info or update_language is True:
            print('\t Detecting language. This step takes a few minutes.')            
            website_info['lang'] = website_info.text.apply(website_text_dataset.detect_lang)
            print("\t Done")

        print("\t .Identify invalid websites")
        website_info = website_text_dataset.add_is_valid_website(website_info)
        print("\t .Done")
        
        print("\t .Identify duplicate websites")
        website_info['is_duplicate'] = website_info.duplicated(subset = ['website','type'])
        print("\t .Done")
        
        return website_info


    

    

    def get_valid_public_firms_index(website_info):
        public_index = np.all([website_info.type == "public_firm",                                
                                 website_info.is_valid_website == True,
                                 website_info.is_duplicate == False],
                                axis=0)

        return public_index


    
    
    def get_valid_startups_index(website_info):
        startups_index = np.all([website_info.type == "startup",
                                 website_info.snapshot_in_window == True,
                                 website_info.is_valid_website == True,
                                 website_info.is_duplicate == False],
                                axis=0)

        return startups_index


        
    
    def add_is_valid_website(website_info):
        website_info['is_valid_website'] = True
        domain_word_pos = website_info.text.str.lower().str.find('domain')
        bad_domain = np.all([domain_word_pos >= 0 , domain_word_pos < 100], axis=0)


        invalid_conditions=[bad_domain,                                   
                            website_info.text.str.contains('BuyDomains.com'),
                            website_info.text.str.contains('This Web page is parked for FREE'),
                            website_info.text.str.contains('Free business profile for provided by Network Solutions'),
                            website_info.text.str.contains('The Sponsored Listings displayed above are served automatically'),

                            website_info.text.str.contains('Apache'),
                            website_info.text.str.contains('website is for sale'),
                            website_info.text.str.contains('This Web site coming soon'),
                            website_info.text.str.contains('Welcome to the new website! Our site has been recently created'),
                            website_info.text.str.match('^Wayback Machine'),
                            website_info.text.str.match('Wayback Machine See what s new with book lending'),
                            website_info.text.str.match('^AVAILABLE NOT FOUND'),
                            website_info.text.str.match('^DefaultHomePage'),
                            website_info.text.str.match('^I?n?t?ernet Archive: Scheduled Mantenance'),
                            website_info.text.str.match('^The page cannot be found'),
                            website_info.text.str.match('^503'),
                            website_info.text.str.match('^5?0?3 Service Unavailable'),
                            website_info.text.str.lower().str.match('domain down'),
                            website_info.text.str.match('^Too Many Requests'),
                            website_info.text.str.match('^Your browser does not support'),
                            website_info.text.str.match('^New Server for COMPANYNAME'),
                            website_info.text.str.contains('this page is parked FREE'),
                            website_info.text.str.contains('domain name was recently registered'),
                            website_info.text.str.contains('placeholder for domain'),
                            website_info.text.str.contains('xtremedata.com  : Low cost'),
                            website_info.text.str.lower().str.contains('domain name registration'),
                            website_info.text.str.contains('Under Construction'),
                            website_info.text.str.contains('This Web site is currently under'),
                            website_info.text.str.contains('This domain name was recently'),
                            website_info.text.str.contains('This page is parked free'),
                            website_info.text.str.match('^Microsoft VBScript runtime error'),
                            website_info.text.str.match('^WebFusion'),
                            website_info.text.str.match('^Register domain names'),
                            website_info.text.str.match('^Moved This page has moved'),
                            website_info.text.str.match('^Coming Soon'),
                            website_info.text.str.contains('Site (is )?Temporarily Unavailable'),
                            website_info.text.str.match('^Under Construction'),
                            website_info.text.str.match('^cPanel'),
                            website_info.text.str.match('^Authorization Required'),
                            website_info.text.str.match('^Top Web Search Directory Top Web Searches'),
                            website_info.text.str.match('^Web Searches'),
                            website_info.text.str.match('^Web Hosting'),
                            website_info.text.str.match('^Search Directory Page Sponsored Listing'),
                            website_info.text.str.match('^coming soon'),
                            website_info.text.str.match('This site is the default web server site'),
                            website_info.text.str.match('DF-1.4 %���� 0 obj< '),
                            website_info.text.str.match('This page uses frames, but your brow'),
                            website_info.text.str.match('U N D E R C O N S T R U C T I O N'),
                            website_info.text.str.match('We recommend you upgrade your browser to one of below free alternatives'),
                            website_info.text.str.match('enable JavaScript'),
                            website_info.text.str.lower().str.match('under construction'),
                            website_info.text.str.match('Page cannot be Please contact your service provider for more'),

                            website_info.text.str.match('^A WordPress Site'),
                            website_info.text.str.match('^Related Searches: Related Searches'),
                            website_info.text.str.match('^Welcome to IIS'),

                            ### Language
                            website_info.lang != 'en']
        
        #Some conditions by range of value in "find" method
        a= website_info.text.str.find("Go Daddy")
        invalid_conditions.append(np.logical_and(a>= 0 , a<200))
        
        a =  website_info.text.str.find("Wayback Machine")
        invalid_conditions.append(np.logical_and(a>= 0 , a<200))
        
        a =  website_info.text.str.find('This website is for sale')
        invalid_conditions.append(np.logical_and(a>= 0 , a<50))
        
        a =  website_info.text.str.find('Adobe Flash Player Download')
        invalid_conditions.append(np.logical_and(a>= 0 , a<30))
        
        website_info.at[np.any(invalid_conditions, axis=0),'is_valid_website'] = False
        
        return website_info


    

    def get_self_index(website_info, i, firmtype="startup"):
        self_index  = np.all([website_info.website == website_info.website[i],
                          website_info.type == firmtype],
                         axis=0)
        return self_index
        
    def get_latest_snapshots(path="../../tfidf/closest_snapshots_list.dta"):
        if os.path.exists(path):            
            print("Using closest snapshots from path {0}".format(path))
            return pd.read_stata(path)
        else:
            print("Could not find a closest snapshot file at {0}".format(path))
            return None
            

    lang_counter = 0
    def detect_lang(text):
        return detect(text)

        
    def setup_website_text_df( website_df, truncate_text_chars=5000):
        #
        #  this is the main method that converts a dataframe of firms into a 
        #  website dataset that can be used to study the similarity across firms.
        #
        
        websites = []        
        website_list= []

        print("Loading all websites. Total: {0}".format(website_df.shape[0]), flush=True)
        counter  = 0
        counter_good = 0

        closest_snapshots = website_text_dataset.get_latest_snapshots()
        snap_year_str = closest_snapshots.closest_snapshot_time.str.slice(0,4)
        closest_snapshots['snapshot_year'] = pd.to_numeric(snap_year_str , errors='coerce')
        closest_snapshots['snapshot_in_window'] = np.absolute(closest_snapshots.founding_year - closest_snapshots.snapshot_year) <= 2
        
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

                if truncate_text_chars is not None:
                    text = text[1:truncate_text_chars] if len(text) > truncate_text_chars else text
                    
                website_info['text'] = text            
                website_info['type'] = row['type']

                close_snap = closest_snapshots[closest_snapshots.website == website_info['website']]


                if close_snap is not None and close_snap.shape[0]  >= 1:
                    website_info['closest_snapshot'] = close_snap.closest_snapshot.iloc[0]
                    website_info['closest_snapshot_time'] = close_snap.closest_snapshot_time.iloc[0]
                    website_info['snapshot_in_window'] = close_snap.snapshot_in_window.iloc[0]
                
                website_list.append(website_info)

            if (counter % 30) == 0:
                print("\t.. {0} ({1})".format(counter, counter_good), flush=True)

        website_info = pd.DataFrame(website_list)    
        return (website_info, websites)
        print("\t Done", flush=True)
        
