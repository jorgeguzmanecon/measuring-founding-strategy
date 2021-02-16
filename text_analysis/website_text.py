import os
from bs4 import BeautifulSoup
import sys
import pdb
import os
import re
sys.path.append(os.path.abspath('../crawler'))
from waybackmachine_crawler import waybackmachine_crawler
sys.path.append(os.path.abspath('../download'))
from data_reader import data_reader


class website_text:

    def __init__(self, path, domain, year, incyear = None, skip_memory_error = False):
        domain = data_reader.clean_domain_url(domain)
        self.texts = self.load_domain(path, domain, year, incyear , skip_memory_error = skip_memory_error)


    def get_website_text(self):
        if self.texts is not None:
            return " ".join(self.texts)
        else:
            return ""

    def clean_page_text(self, text):
        #initial_text = text
        text= re.sub(r"[\s,:#][\.\-@]?[\d\w]+[\.\-@][\d\w\.\-@_]*"," ",text)
        text= re.sub(r"[\s,:][\.\-@][\d\w]+"," ",text)
        text= re.sub(r"\s\d[\d\w]+"," ",text)
        text= re.sub(r"[{}>\*/\';]"," ",text)
        text= re.sub(r"\s+"," ",text)
        text= re.sub(r"DOMContentLoaded.*_static"," ",text)
        return text


    def is_valid_website(self):
        text_len = len(self.get_website_text())

        #this is only the WAybackMachine data
        if text_len == 1706 or text_len < 50:  
            return False

        else:
            return True

        
    def load_page(self, page_file_path, skip_memory_error = False):
        try:
            f = open(page_file_path)
            html = f.read()
            soup = BeautifulSoup(html,"html.parser")
            return self.clean_page_text(soup.get_text(separator = ' '))
        except (TypeError, UnboundLocalError):
            print("\t. --> There was one error, skipping this page")
            return ""
        except MemoryError as e:
            if skip_memory_error:
                print("\t --> Memory Error.. Skipping")
                return ""
            else:
                raise e
    
    def load_domain(self, path, domain, year=None, incyear = None, force_download=False,skip_memory_error = False):

        clean_domain = data_reader.clean_domain_url(domain)
        root_folder = "{0}/{1}".format(path,clean_domain).replace("//","/")

        if year is None:
            file_folder = root_folder
        else: 
            file_folder = "{0}/{1}/{2}".format(path,clean_domain, year).replace("//","/")

        
        
        if  os.path.exists(root_folder) is False or os.path.exists(file_folder) is False or os.path.isdir(file_folder) is False:
            if force_download is True:
                #depends on whether it is startup download of public download
                pdb.set_trace()
                download_year = year if year is not None else incyear
                download_year = int(download_year)
                self.force_download(root_folder , domain, download_year)
            else:
                return None
            
        files = []
        for file_name in os.listdir(file_folder):
            text = self.load_page(file_folder + "/" + file_name, skip_memory_error = skip_memory_error)
            text = re.sub(r"\s+"," ",text)
            files.append(text)

        return files



    
    def force_download(self,path, domain, year):
        year_folder = year is not None

        crawler = waybackmachine_crawler(website = domain, output_folder = path , year_folder = year_folder)
        crawler.crawl_from_date(year, 12, 31)
