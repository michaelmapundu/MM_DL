#Deep learning script
print('importing required packages...')
import pandas as pd, numpy as np, os, re, time, matplotlib.pyplot as plt, seaborn as sns, warnings, psycopg2, json
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

#from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

'''Display'''
from IPython.core.display import display, HTML

from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model


from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import SpatialDropout1D, Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras.layers import Bidirectional
from keras import backend as K
warnings.filterwarnings('ignore')

#seting the columns views to full
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -13)
pd.options.display.float_format = '{:,.2f}'.format
print('required packages loaded and view set successfully...')

#seting a few parameters
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_colwidth', -13)
pd.options.display.float_format = '{:,.2f}'.format
display(HTML("<style>.container { width:95% !important; }</style>"))
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

start = time.time()
startptd = time.strftime('%X %x %Z')
print('\n','The program start time and Date','\n',startptd)

#Setting the working directory
os.chdir('C:/Users/a0056407/Desktop/Michael_Mapundu__Docs/FinalPhDManuscripts2022/DLManuscript/Script/OneDrive-2022-06-09/text only')

print('reading the dataset, please wait...')
conn = psycopg2.connect(user="postgres", password="postgres",
                                  host="127.0.0.1", port="5432", database="mike")
cursor = conn.cursor()
#sql_cmd1 = "select id, disease_description from cleaned_reports"
sql_cmd1 = "select * from verbal_autopsy"
cursor.execute(sql_cmd1)
#cursor.execute(sql_cmd2)
dfA = pd.read_sql(sql_cmd1, conn)
#df2 = pd.read_sql(sql_cmd2, conn)
#closing the connection
cursor.close()
conn.close()
print('data read successfully and connection closed...')

for col in dfA.columns:
    print(col)
    
#drop less important rows
del dfA['dob'], dfA['dod'], dfA['elder'], dfA['midage'], dfA['adult'], dfA['child'], dfA['province'], dfA['womandeath']
del dfA['under5'], dfA['infant'], dfA['neonate'], dfA['deathlocal'], dfA['diedat'], dfA['diedatplace'], dfA['hospital']
del dfA['deathregistration'], dfA['respondentpresent'], dfA['whynorespondent'], dfA['respondentdeceasedrelation']
del dfA['otherrelation'], dfA['famcause1'], dfA['famcause2'], dfA['receiv_biotreatment']
del dfA['biomedi_received'], dfA['receiv_tradi_treatment'], dfA['tradimed_received'], dfA['treatm_sougth_first'], dfA['other_remarks']
del dfA['icdimed_consensus'], dfA['icdcont_consensus'], dfA['maincod_assess1'], dfA['icdmain_assess1'], dfA['lik1'], dfA['lik3']
del dfA['immcod_assess1'], dfA['contrcod_assess1'], dfA['doctor_name_assess1'], dfA['maincod_assess2'], dfA['lik2'], dfA['cause3'] 
del dfA['icdmain_assess2'], dfA['immcod_assess2'], dfA['contrcod_assess2'], dfA['doctor_name_assess2'], dfA['cause2']
del dfA['malprev'], dfA['hivprev'], dfA['pregstat'], dfA['preglik'], dfA['prmat'], dfA['indet'], dfA['cause1']


print('merge the two datasets...')
#dfA = pd.merge(df1, df2, on='id')

print(len(dfA))
print(dfA.head())

print('drop nulls...')
dfA_subA = dfA.dropna(subset=['icdmain_consensus'])
dfA_subA["icdmain_consensus"] = dfA_subA["icdmain_consensus"].str.upper()
dfA_subA['length'] = dfA_subA['icdmain_consensus'].str.len()
#dfA_subA = dfA_subA[dfA_subA.length > 2]
print(len(dfA_subA))
print(dfA_subA.head())
del dfA_subA['length']

print('recod the cause of death categories...')
dfA_subA['broad_cause_cat'] = ''
dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'B20') | (dfA_subA['icdmain_consensus'] == 'B24')|
             (dfA_subA['icdmain_consensus'] == 'A16') | (dfA_subA['icdmain_consensus'] == 'A15'),
             'broad_cause_cat'] = 'HIV/AIDS & TB'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'A40') | (dfA_subA['icdmain_consensus'] == 'A41')|
             (dfA_subA['icdmain_consensus'] == 'J00') | (dfA_subA['icdmain_consensus'] == 'J22')|
             (dfA_subA['icdmain_consensus'] == 'A00') | (dfA_subA['icdmain_consensus'] == 'A09')|
             (dfA_subA['icdmain_consensus'] == 'B50') | (dfA_subA['icdmain_consensus'] == 'B54')|
             (dfA_subA['icdmain_consensus'] == 'B05') | (dfA_subA['icdmain_consensus'] == 'A39')|
             (dfA_subA['icdmain_consensus'] == 'G00') | (dfA_subA['icdmain_consensus'] == 'G05')|
             (dfA_subA['icdmain_consensus'] == 'A33') | (dfA_subA['icdmain_consensus'] == 'A35')|
             (dfA_subA['icdmain_consensus'] == 'A37') | (dfA_subA['icdmain_consensus'] == 'A92')|
             (dfA_subA['icdmain_consensus'] == 'A99') | (dfA_subA['icdmain_consensus'] == 'A90')|
             (dfA_subA['icdmain_consensus'] == 'A91') | (dfA_subA['icdmain_consensus'] == 'A17')|
             (dfA_subA['icdmain_consensus'] == 'A19') | (dfA_subA['icdmain_consensus'] == 'A20')|
             (dfA_subA['icdmain_consensus'] == 'A38') | (dfA_subA['icdmain_consensus'] == 'A42')|
             (dfA_subA['icdmain_consensus'] == 'A89') | (dfA_subA['icdmain_consensus'] == 'B00')|
             (dfA_subA['icdmain_consensus'] == 'B19') | (dfA_subA['icdmain_consensus'] == 'B25')|
             (dfA_subA['icdmain_consensus'] == 'B49') | (dfA_subA['icdmain_consensus'] == 'B55')|
             (dfA_subA['icdmain_consensus'] == 'B99'),'broad_cause_cat'] = 'Other infectious'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'O00') | (dfA_subA['icdmain_consensus'] == 'O03')|
             (dfA_subA['icdmain_consensus'] == 'O08') | (dfA_subA['icdmain_consensus'] == 'O10')|
             (dfA_subA['icdmain_consensus'] == 'O16') | (dfA_subA['icdmain_consensus'] == 'O46')|
             (dfA_subA['icdmain_consensus'] == 'O67') | (dfA_subA['icdmain_consensus'] == 'O72')|
             (dfA_subA['icdmain_consensus'] == 'O63') | (dfA_subA['icdmain_consensus'] == 'O66')|
             (dfA_subA['icdmain_consensus'] == 'O75.3') | (dfA_subA['icdmain_consensus'] == 'O85')|
             (dfA_subA['icdmain_consensus'] == 'O99.0') | (dfA_subA['icdmain_consensus'] == 'O71')|
             (dfA_subA['icdmain_consensus'] == 'P95') | (dfA_subA['icdmain_consensus'] == 'P36')|
             (dfA_subA['icdmain_consensus'] == 'A33') | (dfA_subA['icdmain_consensus'] == 'Q00')|
             (dfA_subA['icdmain_consensus'] == 'Q99') | (dfA_subA['icdmain_consensus'] == 'P00')|
             (dfA_subA['icdmain_consensus'] == 'P04') | (dfA_subA['icdmain_consensus'] == 'P08')|
             (dfA_subA['icdmain_consensus'] == 'P15') | (dfA_subA['icdmain_consensus'] == 'P26')|
             (dfA_subA['icdmain_consensus'] == 'P35') | (dfA_subA['icdmain_consensus'] == 'P37')|
             (dfA_subA['icdmain_consensus'] == 'P94') | (dfA_subA['icdmain_consensus'] == 'P96')|
             (dfA_subA['icdmain_consensus'] == 'P05') | (dfA_subA['icdmain_consensus'] == 'P07')|
             (dfA_subA['icdmain_consensus'] == 'P20') | (dfA_subA['icdmain_consensus'] == 'P22')|
             (dfA_subA['icdmain_consensus'] == 'P23') | (dfA_subA['icdmain_consensus'] == 'P25')|
             (dfA_subA['icdmain_consensus'] == 'O01') | (dfA_subA['icdmain_consensus'] == 'O02')|
             (dfA_subA['icdmain_consensus'] == 'O20') | (dfA_subA['icdmain_consensus'] == 'O45')|
             (dfA_subA['icdmain_consensus'] == 'O47') | (dfA_subA['icdmain_consensus'] == 'O62')|
             (dfA_subA['icdmain_consensus'] == 'O68') | (dfA_subA['icdmain_consensus'] == 'O70')|
             (dfA_subA['icdmain_consensus'] == 'O73') | (dfA_subA['icdmain_consensus'] == 'O84')|
             (dfA_subA['icdmain_consensus'] == 'O86') | (dfA_subA['icdmain_consensus'] == 'O99'),
             'broad_cause_cat'] = 'Martenal & neonatal'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'V01') | (dfA_subA['icdmain_consensus'] == 'V89')|
             (dfA_subA['icdmain_consensus'] == 'V99') | (dfA_subA['icdmain_consensus'] == 'V90')|
             (dfA_subA['icdmain_consensus'] == 'W00') | (dfA_subA['icdmain_consensus'] == 'W19')|
             (dfA_subA['icdmain_consensus'] == 'W65') | (dfA_subA['icdmain_consensus'] == 'W74')|
             (dfA_subA['icdmain_consensus'] == 'X00') | (dfA_subA['icdmain_consensus'] == 'X19')|
             (dfA_subA['icdmain_consensus'] == 'X20') | (dfA_subA['icdmain_consensus'] == 'X29')|
             (dfA_subA['icdmain_consensus'] == 'X40') | (dfA_subA['icdmain_consensus'] == 'X49')|
             (dfA_subA['icdmain_consensus'] == 'X60') | (dfA_subA['icdmain_consensus'] == 'X84')|
             (dfA_subA['icdmain_consensus'] == 'X85') | (dfA_subA['icdmain_consensus'] == 'Y09')|
             (dfA_subA['icdmain_consensus'] == 'X30') | (dfA_subA['icdmain_consensus'] == 'X39')|
             (dfA_subA['icdmain_consensus'] == 'S00') | (dfA_subA['icdmain_consensus'] == 'T99')|
             (dfA_subA['icdmain_consensus'] == 'W20') | (dfA_subA['icdmain_consensus'] == 'W64')|
             (dfA_subA['icdmain_consensus'] == 'W75') | (dfA_subA['icdmain_consensus'] == 'W99')|
             (dfA_subA['icdmain_consensus'] == 'X50') | (dfA_subA['icdmain_consensus'] == 'X59')|
             (dfA_subA['icdmain_consensus'] == 'Y10') | (dfA_subA['icdmain_consensus'] == 'Y98'),
             'broad_cause_cat'] = 'External causes'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'J40') | (dfA_subA['icdmain_consensus'] == 'J44')|
             (dfA_subA['icdmain_consensus'] == 'J45') | (dfA_subA['icdmain_consensus'] == 'J46'),
             'broad_cause_cat'] = 'Respiratory'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'R10') | (dfA_subA['icdmain_consensus'] == 'K70')|
             (dfA_subA['icdmain_consensus'] == 'K76') | (dfA_subA['icdmain_consensus'] == 'N17')|
             (dfA_subA['icdmain_consensus'] == 'N19'), 'broad_cause_cat'] = 'Abdominal'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'G40') | (dfA_subA['icdmain_consensus'] == 'G41'),
             'broad_cause_cat'] = 'Neurological' 

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == '95') | (dfA_subA['icdmain_consensus'] == '99'),
             'broad_cause_cat'] = 'Indeterminate'

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'D50') | (dfA_subA['icdmain_consensus'] == 'D64')|
             (dfA_subA['icdmain_consensus'] == 'E40') | (dfA_subA['icdmain_consensus'] == 'E46')|
             (dfA_subA['icdmain_consensus'] == 'E10') | (dfA_subA['icdmain_consensus'] == 'E14')
             ,'broad_cause_cat'] = 'Metabolic'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'I20') | (dfA_subA['icdmain_consensus'] == 'I25')|
             (dfA_subA['icdmain_consensus'] == 'I60') | (dfA_subA['icdmain_consensus'] == 'I69')|
             (dfA_subA['icdmain_consensus'] == 'D57') | (dfA_subA['icdmain_consensus'] == 'I00')|
             (dfA_subA['icdmain_consensus'] == 'I09') | (dfA_subA['icdmain_consensus'] == 'I10')|
             (dfA_subA['icdmain_consensus'] == 'I15') | (dfA_subA['icdmain_consensus'] == 'I26')|
             (dfA_subA['icdmain_consensus'] == 'I52') | (dfA_subA['icdmain_consensus'] == 'I70')|
             (dfA_subA['icdmain_consensus'] == 'I99'),'broad_cause_cat'] = 'Cardiovascular' 

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'C00') | (dfA_subA['icdmain_consensus'] == 'C06')|
             (dfA_subA['icdmain_consensus'] == 'C15') | (dfA_subA['icdmain_consensus'] == 'C26')|
             (dfA_subA['icdmain_consensus'] == 'C30') | (dfA_subA['icdmain_consensus'] == 'C39')|
             (dfA_subA['icdmain_consensus'] == 'C50') | (dfA_subA['icdmain_consensus'] == 'C51')|
             (dfA_subA['icdmain_consensus'] == 'C58') | (dfA_subA['icdmain_consensus'] == 'C60')|
             (dfA_subA['icdmain_consensus'] == 'C63') | (dfA_subA['icdmain_consensus'] == 'C07')|
             (dfA_subA['icdmain_consensus'] == 'C14') | (dfA_subA['icdmain_consensus'] == 'C40')|
             (dfA_subA['icdmain_consensus'] == 'C49') | (dfA_subA['icdmain_consensus'] == 'C48'),
             'broad_cause_cat'] = 'Neoplasms' 

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'M99') | (dfA_subA['icdmain_consensus'] == 'N00')|
             (dfA_subA['icdmain_consensus'] == 'N16') | (dfA_subA['icdmain_consensus'] == 'N20')|
             (dfA_subA['icdmain_consensus'] == 'N99') | (dfA_subA['icdmain_consensus'] == 'R00')|
             (dfA_subA['icdmain_consensus'] == 'R09') | (dfA_subA['icdmain_consensus'] == 'R11')|
             (dfA_subA['icdmain_consensus'] == 'R94') | (dfA_subA['icdmain_consensus'] == 'D55')|
             (dfA_subA['icdmain_consensus'] == 'D89') | (dfA_subA['icdmain_consensus'] == 'E00')|
             (dfA_subA['icdmain_consensus'] == 'E07') | (dfA_subA['icdmain_consensus'] == 'E15')|
             (dfA_subA['icdmain_consensus'] == 'E35') | (dfA_subA['icdmain_consensus'] == 'E50')|
             (dfA_subA['icdmain_consensus'] == 'E90') | (dfA_subA['icdmain_consensus'] == 'F00')|
             (dfA_subA['icdmain_consensus'] == 'F99') | (dfA_subA['icdmain_consensus'] == 'G06')|
             (dfA_subA['icdmain_consensus'] == 'G09') | (dfA_subA['icdmain_consensus'] == 'G10')|
             (dfA_subA['icdmain_consensus'] == 'G37') | (dfA_subA['icdmain_consensus'] == 'G50')|
             (dfA_subA['icdmain_consensus'] == 'G99') | (dfA_subA['icdmain_consensus'] == 'H00')|
             (dfA_subA['icdmain_consensus'] == 'H95') | (dfA_subA['icdmain_consensus'] == 'J30')|
             (dfA_subA['icdmain_consensus'] == 'J99') | (dfA_subA['icdmain_consensus'] == 'K00')|
             (dfA_subA['icdmain_consensus'] == 'K31') | (dfA_subA['icdmain_consensus'] == 'K35')|
             (dfA_subA['icdmain_consensus'] == 'K38') | (dfA_subA['icdmain_consensus'] == 'K40')|
             (dfA_subA['icdmain_consensus'] == 'K93') | (dfA_subA['icdmain_consensus'] == 'L00')|
             (dfA_subA['icdmain_consensus'] == 'L99') | (dfA_subA['icdmain_consensus'] == 'M00')|
             (dfA_subA['icdmain_consensus'] == 'J39') | (dfA_subA['icdmain_consensus'] == 'J47'),
             'broad_cause_cat'] = 'Other NCD'  


print(dfA_subA['broad_cause_cat'].value_counts())

#keep the two categories
#dfA_subB = dfA_subA[(dfA_subA['broad_cause_cat'] == 'HIV/AIDS & TB') | 
#                    (dfA_subA['broad_cause_cat'] == 'Other infectious')]
dfA_subB = dfA_subA[(dfA_subA['broad_cause_cat'] != '')]

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100


# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(dfA_subB['disease_description'].values.astype(str))
X = tokenizer.texts_to_sequences(dfA_subB['disease_description'].values.astype(str))
X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
#creating dummy labels
#Y = pd.get_dummies(dfA_subB['broad_cause_cat']).values
print('encoding the labels in the dataset, please wait...')
#Turning labels into numbers
encoder = LabelBinarizer()
encoder.fit(dfA_subB['broad_cause_cat'])
y = encoder.transform(dfA_subB['broad_cause_cat'])
y1 = np.argmax(y, axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.30, random_state = 0)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

df = pd.DataFrame(X)
df['target'] = y1
plt.figure(figsize=(12, 8))
df.target.value_counts().plot(kind='bar', title='Count (target)')
plt.ylabel('Counts', fontsize = 13)
plt.xlabel('classes', fontsize = 13)
plt.savefig('classes_before_balancing.png', dpi = 900)
plt.savefig('classes_before_balancing.pdf', dpi = 900)
plt.show()

print('balancing the data')
# transform the dataset
strategy = {0:3388, 1:3388, 2:3388, 3:3388, 4:3388, 5:3388, 6:3388, 7:3388, 8:3388, 9:3388, 10:3388, 11:3388}
oversample = SMOTE(sampling_strategy=strategy)
X, y = oversample.fit_resample(X, y)
y1 = np.argmax(y, axis = 1)

#smote = SMOTE('minority')
#X, y = smote.fit_sample(X, y)

df = pd.DataFrame(X)
df['target'] = y1
plt.figure(figsize=(12, 8))
df.target.value_counts().plot(kind='bar', title='Count (target)')
plt.ylabel('Counts', fontsize = 13)
plt.xlabel('classes', fontsize = 13)
plt.savefig('classes_after_balancing.png', dpi = 900)
plt.savefig('classes_after_balancing.pdf', dpi = 900)
plt.show()

print('inserting embedding file')
embeddings_index = {}
glove_data = 'C:/Users/a0056407/Desktop/Michael_Mapundu__Docs/FinalPhDManuscripts2022/DLManuscript/Script/OneDrive-2022-06-09/text only/glove.6B/glove.6B.100d.txt'
f = open(glove_data, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

num_words = min(MAX_NB_WORDS, len(word_index)) + 1
print(num_words)

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        


print('Build lstm model...')
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


print('Train...')
epochs = 50
batch_size = 128

hist = model.fit(X_train, Y_train, epochs=epochs, 
                 batch_size=batch_size,validation_split=0.2,
                 callbacks=[EarlyStopping(monitor='val_loss', 
                                          patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])
axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.savefig('accuracy_measure_lstm.png', dpi = 900)
plt.savefig('accuracy_measure_lstm.pdf', dpi = 900)
plt.show()

print("Validation_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(Y_test))[1]*100))
print("Training_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_train), np.array(Y_train))[1]*100))


#dfB = pd.concat([dfB_sub['id'],dfB_sub['disease_description']], axis=1)
dfA['disease_description'] = dfA['disease_description'].astype(str)
print('Creating padded sequence...')
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(dfA['disease_description'])
sequences_clasify = tokenizer.texts_to_sequences(dfA['disease_description'])
padded = sequence.pad_sequences(sequences_clasify, maxlen=MAX_SEQUENCE_LENGTH)
#pred = model.predict(padded, batch_size=batch_size, verbose=1)
print('Performing the actual classification for lstm...')
#dfB['pred_lstm'] = model.predict_classes(padded, batch_size=batch_size, verbose=1)
dfA['pred_lstm'] = np.argmax(model.predict(padded, batch_size=batch_size),axis = 1)

print('Build cnn model...')
model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
#model1.add(Conv1D(512, 3, activation='relu'))
#model1.add(MaxPooling1D(3))
#model1.add(Conv1D(256, 3, activation='relu'))
#model1.add(MaxPooling1D(3))
model1.add(Conv1D(64, 3, activation='relu'))
model1.add(MaxPooling1D(3))
model1.add(Flatten())
model1.add(Dense(64, activation='relu'))
model1.add(Dense(12, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()

hist1 = model1.fit(X_train, Y_train, epochs=epochs, 
                 batch_size=batch_size,validation_split=0.2,
                 callbacks=[EarlyStopping(monitor='val_loss', 
                                          patience=3, min_delta=0.0001)])

accr1 = model1.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr1[0],accr1[1]))

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist1.epoch, hist1.history['loss'])
axs[0].plot(hist1.epoch, hist1.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist1.epoch, hist1.history['accuracy'])
axs[1].plot(hist1.epoch, hist1.history['val_accuracy'])
axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.savefig('accuracy_measure_cnn.png', dpi = 900)
plt.savefig('accuracy_measure_cnn.pdf', dpi = 900)
plt.show()

print("Validation_Accuracy: {:.2f}%".format(model1.evaluate(np.array(X_test), np.array(Y_test))[1]*100))
print("Training_Accuracy: {:.2f}%".format(model1.evaluate(np.array(X_train), np.array(Y_train))[1]*100))

print('Performing the actual classification for cnn...')
dfA['pred_cnn'] = model1.predict_classes(padded, batch_size=batch_size, verbose=1)


conn = psycopg2.connect(user="postgres", password="postgres",
                                  host="127.0.0.1", port="5432", database="mike")

print('deleting table from database if exist')
with conn:
    cursor = conn.cursor()
    cursor.execute('drop table if exists dl_using_text')
    conn.commit()

print('exporting the dataframe to postgres table...')
engine = create_engine('postgresql://postgres:postgres@localhost:5432/mike')
dfA.to_sql('dl_using_text', engine, chunksize=1000, index=False)
print('data loaded to postgres successfully...') 

print('halting...') 
stoptd = time.strftime('%X %x %Z')
print('\n','The program stop time and Date','\n',stoptd)
print('It took', ((time.time()-start)/60)/60, 'hours to run the script.')

print('self clean up...')

print('complete...')  
