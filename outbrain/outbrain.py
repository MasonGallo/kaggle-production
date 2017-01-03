import pandas as pd
import numpy as np

# dtypes based on value ranges from bash script - this saves memory cost
dtypes_train = {'ad_id': np.uint32, 'clicked': np.int8}
train = pd.read_csv("data/clicks_train.csv", usecols=['ad_id','clicked'],
    dtype=dtypes_train)

# value_counts should be faster than using groupby
all_ads = train['ad_id'].value_counts()
clicked = train[(train.clicked == 1)]['ad_id'].value_counts()
overall_ctr = train.clicked.mean()
combo = pd.concat([clicked, all_ads], axis=1)
combo.columns = ['clicks', 'imps']
combo.fillna(0, inplace=True)

# we don't need train anymore so let's save memory
del train

# next we compute the ctr per ad but add a regularization term in order to
# account for ads that may have only been served a few times or many times
reg = 10
combo['ctr'] = (combo['clicks'] + reg*overall_ctr) / (combo['imps'] + reg)
combo = combo.reset_index()
combo.drop(['clicks', 'imps'], axis=1, inplace=True) # save memory
combo.columns = ['ad_id', 'ctr']

# based on value range from bash script to save memory
dtypes_test = {'ad_id': np.uint32, 'display_id': np.uint32}
test = pd.read_csv("data/clicks_test.csv", dtype=dtypes_test)
test = test.merge(combo, how='left')
test.ctr.fillna(overall_ctr, inplace=True) # if we haven't seen, guess overall
test.sort_values(['display_id','ctr'], inplace=True, ascending=False)
test.drop('ctr', axis=1, inplace=True)

# convert to format appropriate for submission
subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
subm.to_csv("subm.csv", index=False)
