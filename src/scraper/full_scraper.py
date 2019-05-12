#!/Users/julieshih/anaconda/bin/python

import requests
import urllib.request
import pandas as pd
import urllib3
from bs4 import BeautifulSoup
import logging
import html 
import time
import os 

#logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG)

clothing_list = ['top_blouses', 'dress']
pageno = range(1,2)


# function parses the response for data on each clothing item
def get_links(res):

    soup = BeautifulSoup(res.text, 'html.parser')
    
    item_urls = []
    img_urls = []
    item_desc = []
    price_list = []
    for_sale = []
    full_price = []

    items = soup.find_all('a', class_='item_slider product_link')

    for item in items:
        for img in item.find_all('img', class_='product_image represent lazyload'):
            #save only default img, ignore other views
            if 'default' in img['data-original']:                           
                if 'default' not in img['alt']:
                    item_urls.append('https://www.forever21.com'+item['href'])  
                    item_desc.append(img['alt'])                            #image name
                    img_urls.append(img['data-original'])                      
            else:
                pass    

    prices = soup.find_all('p', class_='p_price')


    for price in prices:

        pr = price.getText().strip().replace('$','')

        if 'OFF' in pr:
            price_list.append(pr.split('\n')[0])
            full_price.append(pr.split('\n')[1])
            for_sale.append('True')
        else:
            price_list.append(pr)
            full_price.append(pr)
            for_sale.append('False')

    logging.info(len(item_urls), len(img_urls), len(item_desc), len(price_list), len(full_price))
    assert len(item_urls) == len(img_urls) == len(item_desc) == len(price_list) == len(full_price)
    return item_urls, img_urls, item_desc, price_list, for_sale, full_price


# loops through pages and saves data as csv per clothing type
for clothing_type in clothing_list:

    for page in pageno:
    
        logging.info('retrieving '+clothing_type+', page '+ str(page) + '...'  ) 
    
        url = 'https://www.forever21.com/us/shop/catalog/category/f21/'+clothing_type+'#pageno='+str(page)
        res = requests.get(url)
    
        item_urls, img_urls, item_desc, price_list, for_sale, full_price = get_links(res)

        # create dataframe of scraped info
        df = pd.DataFrame(
            {
            'url': item_urls,
            'img' : img_urls,
            'name' : item_desc,
            'prices': price_list,
            'for_sale': for_sale,
            'full_price': full_price
            } )

        # add date to dataframe
        df['date_scraped'] = str(time.strftime("%Y-%m-%d"))

        # add path to image
        df['path'] = 'img/'+df.img.str.split('/').str[-1]

        # make directory for clothing type
        directory = 'f21/'+clothing_type
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory+'/'+img)
        
        name = directory+'/'+clothing_type+'_page'+str(page)+'.csv'
        df.to_csv(name, index=False)
    
        time.sleep(5)

