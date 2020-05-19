import pandas as pd
import numpy as np
import argparse

"""
cleaning of scraped data
"""
def data_cleaning(dataset="dataset.csv",cleaned_dataset="cleaned_dataset.csv"):
	# reading scrap data

	df=pd.read_csv(dataset)

	# removing uncessory column 
	df.drop(['Unnamed: 0'],axis=1,inplace=True)

	data=df.copy()
	data.reset_index(drop=True,inplace=True)

	# first part of the name repesent 'brand'
	data['Brand']=data['name'].apply(lambda x: x.split()[0])


	data['price']=data['price'].apply(lambda x:x.replace('₹','').replace(',',''))

	# Not decided to seperate the all netwroking config as most of dataset have pretty common networking configuration 
	data['Networking']=data['features'].apply(lambda x:x.split("'")[1])
	data['rating']=data['rating'].apply(lambda x:x.replace('User ratings: ','').split('/')[0])

	# Seperating out different list element of feature which contains all the feature data.
	data['feature_2']=data['features'].apply(lambda x:x.split("'")[3].replace('u2009',' '))
	data['feature_3']=data['features'].apply(lambda x:x.split("'")[5].replace('u2009',' '))
	data['feature_4']=data['features'].apply(lambda x:x.split("'")[7].replace('u2009',' '))
	data['feature_5']=data['features'].apply(lambda x: x.split("'")[9].replace('u2009',' ') if (len(x.split("'"))>9)  else False )
	data['feature_6']=data['features'].apply(lambda x: x.split("'")[11].replace('u2009',' ') if (len(x.split("'"))>11)  else False )
	data['feature_7']=data['features'].apply(lambda x: x.split("'")[13].replace('u2009',' ') if (len(x.split("'"))>13)  else False )
	data['feature_8']=data['features'].apply(lambda x: x.split("'")[15].replace('u2009',' ') if (len(x.split("'"))>15)  else False )

	# Extracting diffent features which orginally embedded within a single list.
	data['Processor(GHz)']=data['feature_2'].apply(lambda x:x.split(',')[2].split('\\')[0].replace(' ','') if len(x.split(','))==3 else np.nan)
	data['Processor']=data['feature_2'].apply(lambda x:x.split(',')[0] if len(x.split(','))==3 else np.nan)
	data['Core']=data['feature_2'].apply(lambda x:x.split(',')[1] if len(x.split(','))==3 else np.nan)
	data['RAM(GB)']=data['feature_3'].apply(lambda x:x.split(',')[0].replace('\\','').split()[0])
	data['Memory(GB)']=data['feature_3'].apply(lambda x:x.split(',')[1].replace('\\','').split()[0] if len(x.split(','))>1 else np.nan)
	data['Battery']=data['feature_4'].apply(lambda x:x.split(',')[0].split('\\')[0] )
	data['Screen_size(inches)']=data['feature_5'].apply( lambda x:x.split(',')[0].split(' ')[0])
	data['Display']=data['feature_5'].apply(lambda x:x.split(',')[1].replace('\\','') if len(x.split(','))==2 else np.nan)
	data['Rear_cam']=data['feature_6'].apply(lambda x:x.split('&')[0].replace('\\','').replace('MP','').split('+') if x!=False else np.nan)
	data['Front_cam']=data['feature_6'].apply(lambda x:x.split('&')[1].replace('\\','').replace('MP','').split('+') if x!=False and len(x.split('&'))==2 else np.nan)
	data['Memory_card_supported']=data['feature_7'].apply(lambda x:x.split('\\')[0].split(' ')[-1] if x!=False else np.nan)
	data['Android-v']=data['feature_8']

	# droping embedded lists after getting all the features out of it.
	data.drop(['features','feature_2','feature_3','feature_4'	,'feature_5','feature_6','feature_7','feature_8'],axis=1,inplace=True)
	data['rating']=data['rating'].apply( lambda x:x if len(x)<6 else np.nan)
	data.to_csv(cleaned_dataset)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset',help="address of original dataset with format '.csv' ")
	parser.add_argument('--cleaned_dataset',help="address of final dataset with format '.csv' ")
	
	dataset=parser.dataset
	cleaned_dataset=parser.cleaned_dataset

	data_cleaning(dataset,cleaned_dataset)


