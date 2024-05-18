import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import string
import numpy as np

captions_df = pd.read_csv('results.csv', sep='|', index_col=False)
#print(captions_df.head())

#stripping white space from column name
captions_df.columns = captions_df.columns.str.strip()  
captions_df.drop(['comment_number'], axis = 1,inplace=True) 
#print(captions_df.head())

captions_df.rename(columns = {'comment':'caption'}, inplace = True)
#print(captions_df.head())


# save_captions_df = captions_df.drop(['image_name'], axis = 1)
# save_captions_df.head()

# np.savetxt(r'temp_captions.txt', captions_df.values, fmt='%s', delimiter='\t')
# np.savetxt(r'temp_captions_new.txt', save_captions_df.values, fmt='%s', delimiter='\t')


captions_df['caption'] = captions_df['caption'].str.replace('\d+\.\d+', '')
# print(captions_df.head())



# save_captions_df1 = captions_df.drop(['image_name'], axis = 1)
# np.savetxt(r'temp_captions_newnew.txt', save_captions_df1.values, fmt='%s', delimiter='\t')



def clean_text(text):
    #function for cleaning the captions,
    #removing punctuations, converting all to lowecase
    #etc.
    
    table = str.maketrans('','',string.punctuation)
    text = str(text)
    text = text.replace("-", " ")
    text = text.lower()
    text = text.translate(table)
    
    return text


captions_df['caption'] = captions_df['caption'].apply(clean_text)
print(captions_df.head())
# print(captions_df.info())

##################################################################################################################################
#np.savetxt(r'descriptions.txt', captions_df.values, fmt='%s', delimiter='\t')
##################################################################################################################################


# Building vocabulary of all unique words
vocab = set()
for each_caption in captions_df["caption"]:
    vocab.update(each_caption.split())
        
print("length of vocabulary = ", len(vocab))
#print(vocab)



dev_df = pd.DataFrame(captions_df["image_name"])
dev_df.drop_duplicates(subset ="image_name",keep = "first", inplace = True)
#print(dev_df.head())
#print(dev_df.info())

dev_df = dev_df.reset_index(drop=True)
#print(dev_df.head())
#print(dev_df.info())


##########################################################################################################
# For train, validate and test set creation
def train_validate_test_split(df, train_percent=.92, validate_percent=.04, seed=43):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return (train, validate, test)



train, validate, test = train_validate_test_split(dev_df)

print("Training Details")
print(train.head())
print(len(train))
print()

print("Validation Details")
print(validate.head())
print(len(validate))
print()

print("Test Details")
print(test.head())
print(len(test))
print()

print("---------------------------------------------------------------------------------------------------------------------")
train = train.reset_index(drop=True)
print(train.head())

validate = validate.reset_index(drop=True)
print(validate.head())

test = test.reset_index(drop=True)
print(test.head())


np.savetxt(r'Flickr_30k.trainImages.txt', train.values, fmt='%s', delimiter='\t')  # Training Set   - 25,426 images
np.savetxt(r'Flickr_30k.devImages.txt', validate.values, fmt='%s', delimiter='\t') # Validation Set - 3,178 images
np.savetxt(r'Flickr_30k.testImages.txt', test.values, fmt='%s', delimiter='\t')    # Testing Set    - 3,179 images
