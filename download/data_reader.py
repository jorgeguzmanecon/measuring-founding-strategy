import pandas as pd
import pdb
import os, re
import numpy as np

class data_reader:


    def clean_domain_url(website):
        s =  re.sub("www\.","",website)
        s =  re.sub("home\.","",s)
        s  = re.sub(r"\:.*","",s)
        s = re.sub(r"http(s?)://(\w+)\.(\w+)\.(\w+)",r"\3.\4",s)
        return s



    def add_closest_snapshot(companies, closest_snapshot_path="../../tfidf/closest_snapshots_list.dta"):
        if not os.path.exists(closest_snapshot_path):
            return companies

        snaps = pd.read_stata(closest_snapshot_path)
        snaps = snaps[['org_uuid','closest_snapshot','closest_snapshot_time']]
        snaps['closest_snapshot_year'] = pd.to_numeric(snaps.closest_snapshot_time.str.slice(stop=4), errors='coerce', downcast='integer')

        
        companies =  companies.merge(snaps, how='left', on='org_uuid')
        companies['snapshot_in_window'] = np.absolute(companies.closest_snapshot_year - companies.founding_year) <= 2
        return companies

    def read_crunchbase():
        companies = pd.read_stata("../../Data/Crunchbase/crunchbase_orgs.dta")       
        
        if not "closest_snapshot_time" in companies.columns:
            companies = data_reader.add_closest_snapshot(companies)

        return companies

    
    def read_preqin():
        companies = pd.read_stata("../../Data/Preqin/all_deals.minimal.dta")
        return companies

    def read_public_companies(regex = None):
        companies = pd.read_stata("../../Data/public companies/public_firm_websites.dta")
        companies = companies.rename(columns = {'websiteaddress':'website'})

        if regex is None:
            return companies
        else:
            return companies[companies.website.str.contains(regex)]



    def find_missing_public(year):
        public_firms = data_reader.read_public_companies()
        public_firms = public_firms[public_firms.ipoyear <= int(year)]
        public_firms['year'] = year
        public_firms['in_data'] = False
        
        path = "../../out_public/"
        for i in public_firms.index:        
            domain = public_firms.loc[i,'website']
            domain = data_reader.clean_domain_url(domain)

            file_folder = "{0}/{1}/{2}".format(path,domain, year)            
            if   os.path.exists(file_folder):
                public_firms.loc[i,'in_data'] = True
    
        missing_firms =  public_firms[~public_firms.in_data]
        
        print ("Finished checking for missing public firms in year" +year)
        print ("Report on Missing. \n \t. {0} Total public websites.\n\t. {1} downloaded.\n\t. {2} missing.".format(public_firms.shape[0], public_firms.shape[0] - missing_firms.shape[0], missing_firms.shape[0]))

        return missing_firms
