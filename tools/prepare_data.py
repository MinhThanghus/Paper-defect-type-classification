

import csv
import os
import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# step 2:
# path =r""  #path to orinal data train
# result_test = "fold_data_original.csv"
# for filename in os.listdir(path):
#     for image_ime in os.listdir(path+str("/")+str(filename)):
#         path_image_all =path+str("/")+str(filename)+str("/")+image_ime
#         line = path_image_all,filename
#         with open(result_test, "a", encoding="utf-8", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(line)
# step 2:
N_FOLDS = 5
MULTI = True
NUM_FRAMES = 3

df = pd.read_csv(r'./fold_data_original.csv')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    df.loc[val_idx, 'fold'] = i
df.to_csv("fold_data.csv")
# Step 3:
import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


df = pd.read_csv(r'fold_data.csv')
path_fold ="" #Path of fold data train or val (fold1,2,3,4,5)
for id, row in tqdm(df.iterrows()):
    # image_id = row['id'].split('.')[0]
    label = row['label']
    fold = row['fold']
    path = row['path']
    import os
    if fold ==1 or fold ==2:
    # if fold !=1 and fold !=2:
        directory = path_fold + str("/") + label
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copy(path, directory)
