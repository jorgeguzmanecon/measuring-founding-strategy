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

public_output_folder = "/shared/share_scp/startup_strategy/out_class"


crawler = waybackmachine_crawler("www.tubemogul.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2006,1,1)

crawler = waybackmachine_crawler("www.dachisgroup.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2008,1,1)

crawler = waybackmachine_crawler("www.barefruitsnacks.com", output_folder=public_output_folder)   
crawler.crawl_from_date(2001,1,1)
crawler = waybackmachine_crawler("www.homeaway.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2005,1,1)

crawler = waybackmachine_crawler("www.transactis.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2007,1,1)

crawler = waybackmachine_crawler("www.evestment.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2000,1,1)

crawler = waybackmachine_crawler("www.ethertronics.com", output_folder=public_output_folder)      
crawler.crawl_from_date(2000,1,1)

crawler = waybackmachine_crawler("www.fyrestormic.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2002,1,1)

crawler = waybackmachine_crawler("www.global-analytics.com", output_folder=public_output_folder)  
crawler.crawl_from_date(2003,1,1)

crawler = waybackmachine_crawler("www.venafi.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2000,1,1)

crawler = waybackmachine_crawler("www.blurb.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2005,1,1)

crawler = waybackmachine_crawler("www.lendio.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2011,1,1)

crawler = waybackmachine_crawler("www.privateerholdings.com", output_folder=public_output_folder)  
crawler.crawl_from_date(2011,1,1)

crawler = waybackmachine_crawler("www.immunomix.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2005,1,1)

crawler = waybackmachine_crawler("www.wanelo.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2012,1,1)

crawler = waybackmachine_crawler("www.philzcoffee.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2003,1,1)

crawler = waybackmachine_crawler("www.kinetic-social.com", output_folder=public_output_folder)     
crawler.crawl_from_date(2011,1,1)

crawler = waybackmachine_crawler("www.prognosisinnovation.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2006,1,1)

crawler = waybackmachine_crawler("www.cerusendo.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2011,1,1)

crawler = waybackmachine_crawler("www.activerain.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2004,1,1)

crawler = waybackmachine_crawler("www.momtrusted.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2011,1,1)

crawler = waybackmachine_crawler("www.zagster.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2007,1,1)

crawler = waybackmachine_crawler("www.catchfree.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2010,1,1)

crawler = waybackmachine_crawler("www.redislabs.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2011,1,1)

crawler = waybackmachine_crawler("www.theivorycompany.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2013,1,1)

crawler = waybackmachine_crawler("www.playnomics.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2009,1,1)

crawler = waybackmachine_crawler("www.brightbytes.net", output_folder=public_output_folder)        
crawler.crawl_from_date(2012,1,1)

crawler = waybackmachine_crawler("www.phonezoo.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2006,1,1)

crawler = waybackmachine_crawler("www.jumpbikes.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2010,1,1)

crawler = waybackmachine_crawler("www.aquaspy.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2010,1,1)

crawler = waybackmachine_crawler("www.tabtor.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2010,1,1)

crawler = waybackmachine_crawler("www.padlet.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2012,1,1)

crawler = waybackmachine_crawler("www.liquor.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2008,1,1)

crawler = waybackmachine_crawler("www.dormify.com", output_folder=public_output_folder)        
crawler.crawl_from_date(2011,1,1)
