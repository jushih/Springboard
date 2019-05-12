#!/Users/julieshih/anaconda/bin/python
import os
import sys
import pandas as pd

# file accepts path to images as input and outputs a csv of all filenames

path = sys.argv[1]
output_path = sys.argv[2] # where to save csv

print('Crawling', path , '...' )
#path = '/content/gdrive/My Drive/data/img/'
#output+path = '/content/gdrive/My Drive/img_attributes/Anno/paths.csv'

def get_files(root_dir):
  f = []
  for root, _, filenames in os.walk(path):
    for filename in filenames:
      f.append(os.path.join(root, filename))

  return f

files = get_files(path)

paths_df = pd.DataFrame({'files':files})
paths_df.to_csv(output_path+'paths.csv',index=False)
print('Saved results to ',output_path+'paths.csv')
