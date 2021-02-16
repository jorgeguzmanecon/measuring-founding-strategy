from waybackmachine_crawler import waybackmachine_crawler
import pdb

x = waybackmachine_crawler('www.lytro.com')
x.crawl_from_date(2013,01,01)
