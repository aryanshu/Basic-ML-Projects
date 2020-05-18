from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as ureq
import requests 

import pandas as pd
df = pd.DataFrame(columns=['pid','score-val','name','features','price','rating'])

def get_details(containers , j):
	for i,container in enumerate(containers):
		try:
			val= container.div.div.text
			name  = container.h2.a.text

			feature_all=[]
			product_features = container.find('ul',{"class":"pros"})
			product_features = product_features.find_all('span')
			for feature in product_features:
				feature_all.append(feature.text)

			product_features = container.find('ul',{"class":"cons"})
			product_features = product_features.find_all('span')
			for feature in product_features:
				feature_all.append(feature.text)

			features  =feature_all

			extra = container.find('div',{'class':'extra'})
			price =extra.span.text



			rating = extra.div.span['title']
			pid = extra.find('div',{'class':"button compare"})['pid']
			ii=i+j
			df.loc[ii]=[pid ,val , name, feature_all ,  price , rating ]
		except:
			pass

def get_soup(my_url , i=0):
	page = requests.get(my_url)
	page_html = page.text
	page_soup = soup(page_html , 'html.parser')
	containers = page_soup.findAll("li",{"class":"f-mobiles"})
	get_details(containers, i)

for page in range(1,100):
  r= 'https://www.smartprix.com/mobiles/?page='+str(page)
  i=(page-1)*25
  get_soup(r , i)
df.to_csv('dataset.csv')