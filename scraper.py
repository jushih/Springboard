#!/Users/julieshih/anaconda/bin/python

import requests
import pandas as pd
import urllib3
from bs4 import BeautifulSoup
import logging
import html 
import time
import os 
import urllib.request
import pdb

#logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG)

clothing_list = ['top_blouses']
pageno = range(5,11)

# function parses the response for data on each clothing item
def get_imgs(res, img_dir, page):

    soup = BeautifulSoup(res.text, 'html.parser')
    
    item_urls = []
    img_urls = []
    item_desc = []
    img_paths = []

    items = soup.find_all('a', class_='item_slider product_link')

    for item in items:
        for img in item.find_all('img', class_='product_image represent lazyload'):
            #save only default img, ignore other views
            if 'default' in img['data-original']:                           
                if 'default' not in img['alt']:
                    item_urls.append('https://www.forever21.com'+item['href'])  
                    img_urls.append(img['data-original'])                      
                    item_desc.append(img['alt'])

                    img_path = img_dir+'page'+str(page)+'_'+img['data-original'].split('/')[-1]
                    img_paths.append(img_path)

#                    pdb.set_trace()
                    try:
                        urllib.request.urlretrieve(img['data-original'], img_path)
                        time.sleep(3)
                    except:
                        print('cannot retrieve ' + img['data-original'])
            else:
                pass    


    assert len(item_urls) == len(img_urls) == len(item_desc) 
    return item_urls, img_urls, item_desc, img_paths


# loops through pages and saves data as csv per clothing type
for clothing_type in clothing_list:

    # make directory for clothing type
    directory = 'f21/'+clothing_type
    img_dir = directory+'/img/'

    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(img_dir)
        

    for page in pageno:
    
        logging.info('retrieving '+clothing_type+', page '+ str(page) + '...'  ) 
    
        url = 'https://www.forever21.com/us/shop/catalog/category/f21/'+clothing_type+'#pageno='+str(page)
        res = requests.get(url)
    
        item_urls, img_urls, item_desc, img_paths  = get_imgs(res, img_dir, page)

        # create dataframe of scraped info
        df = pd.DataFrame(
            {
            'url': item_urls,
            'img' : img_urls,
            'name' : item_desc,
            'path' : img_paths,
            } )

        # add date to dataframe
        df['date_scraped'] = str(time.strftime("%Y-%m-%d"))
        name = directory+'/'+clothing_type+'_page'+str(page)+'.csv'
        df.to_csv(name, index=False)
    
        time.sleep(5)

