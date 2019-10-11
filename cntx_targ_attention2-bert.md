---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Setting up the modules 

```python
# common libraries
import sys, glob, random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
# analysis related modules
from scipy.stats import entropy, spearmanr
from scipy import interp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, roc_curve, auc
from sklearn.model_selection import KFold, GroupKFold
from collections import Counter
```

```python
# keras modules
import tensorflow as tf
# from keras.models import load_model

from bert.tokenization import FullTokenizer

# from keras.preprocessing.text import text_to_word_sequence, Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# from keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.utils import plot_model
import keras_tqdm
from livelossplot.keras import PlotLossesCallback
```

```python
# custom layers from external files
from layers.embeddings import BertLayer
from layers.attention import AttentionLayer
from models.build_models import build_model_bert, initialize_vars
from models.train_models import *
```

```python
# some option settings for Jupyter notebook and TF
%load_ext autoreload
%autoreload 2
pd.set_option('display.max_colwidth', -1)
tf.logging.set_verbosity(tf.logging.ERROR)
```

<!-- #region {"toc-hr-collapsed": false} -->
# import cloze sentences 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 8769, "status": "ok", "timestamp": 1569434161204, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="lcI5I1AQI2Df" outputId="141abf35-98ea-448a-8c99-241a15d22498"
try:
    from google import colab
    IN_COLAB = True
    # mount GoogleDrive for data and external resources
    colab.drive.mount('/content/drive')

    # download and install additional libraries
    !pip install keras_tqdm -q
    !pip install livelossplot -q
    
    
#     sys.path.append('./drive/My Drive/Colab Notebooks')  # append colab directory to Python kernel's path
    df_cloze = pd.read_pickle('./drive/My Drive/Colab Notebooks/dataset/cloze_df_scores_all3.pickle')
except:
    IN_COLAB = False
    df_cloze = pd.read_pickle('./dataset/cloze_df_scores_all3.pickle') # enter the local location of DSCoVAR sentence data
```

<!-- #region {"colab_type": "text", "id": "PCwwojkfL6_l"} -->
## preprocessing model inputs
<!-- #endregion -->

```python
# some constants
RDM_SEED = 1
K_FOLDS = 5
MAX_SEQ_LEN = 30
```

```python

```

```python
def proc_sentences(df, col_sentence, col_targ):
    sentences = []
    li_targ = []
    li_sent = []
    for i in range(df.shape[0]):
        sent = df[col_sentence][i]
        targ = None
        if(col_targ):
            targ = df[col_targ][i]
        else:
            targ = "[MASK]"
        sent = sent.replace("______", targ)
        li_targ.append(targ)
        li_sent.append(sent)
    sentences = [np.array(li_targ), np.array(li_sent)]
    return(sentences)

sentences = proc_sentences(df_cloze, 'sentence', 'targ')
```

```python
# sentences_wttarg = proc_sentences(df_cloze, 'sentence', 'syn1')
# sentences_notarg = proc_sentences(df_cloze, 'sentence', None)
sentences = proc_sentences(df_cloze, 'sentence', 'targ')
```

```python
sentences
```

```python

```

```python colab={} colab_type="code" id="IZUg96JlhwqB"
mm_scaler = MinMaxScaler()

resp_scores = mm_scaler.fit_transform(df_cloze[['ent_cloze', 
                                                'elmo_score', 'bert_score', 'glove_score',
                                                'scores_sum', 'sent_len']])
resp_lex = resp_scores[:, 0]
resp_lex = 1-resp_lex # reversing the direction: high score for high informative sentences

resp_brt = resp_scores[:, 1]
resp_lmo = resp_scores[:, 2]
resp_glv = resp_scores[:, 3]
resp_bws = resp_scores[:, 4]
sent_len = resp_scores[:, 5]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 566} colab_type="code" executionInfo={"elapsed": 14111, "status": "ok", "timestamp": 1569434167349, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="dMBnF8u4RUm9" outputId="8ba84efe-5ea6-4f04-829f-797a89d9c9e7"
sns.pairplot(pd.DataFrame({"resp_lex":resp_lex, 
                           "resp_lmo":resp_lmo, "resp_brt":resp_brt, "resp_glv":resp_glv, 
                           "resp_bws":resp_bws,
                           "sent_len":sent_len}))
```

```python
tokenizer = create_tokenizer_from_hub_module()
train_examples = convert_text_to_examples(sentences[0], sentences[1], resp_bws)
tt = convert_examples_to_features(tokenizer, train_examples[:1], True, MAX_SEQ_LEN)
tt[:4]
```

<!-- #region {"colab_type": "text", "id": "l4rgsjFnxP3U", "toc-hr-collapsed": false} -->
# BERT + Attention model
<!-- #endregion -->

```python
K.clear_session()
sess = tf.Session()

model = build_model_bert(MAX_SEQ_LEN, attention_layer=True)
initialize_vars(sess)

model.summary()
```

```python
plot_model(model)
```

# K-fold training and predictions 

```python
# target word location categories
targ_loc_cat = []
for i in range(df_cloze.shape[0]):
    if(df_cloze['targ_loc_end'][i]): 
        targ_loc_cat.append('3_last')
    else:
        if(df_cloze['targ_loc_rel'][i] <= 0.5):
            targ_loc_cat.append('0_less_50')
        elif((df_cloze['targ_loc_rel'][i] > 0.5)&(df_cloze['targ_loc_rel'][i] <= 0.65)):
            targ_loc_cat.append("1_(50_65]")
        elif((df_cloze['targ_loc_rel'][i] > 0.65)):
            targ_loc_cat.append("2_more_65")            
```

```python
Counter(targ_loc_cat)
```

```python
sent_len_cat = pd.qcut(sent_len, [0, 0.20, 0.40, 0.60, 0.80, 1])
sent_len_cat.value_counts()
```

```python
_num_iter = 15
_batch_size = 64

# fold settings
gkf1 = GroupKFold(n_splits=K_FOLDS) ## target words
gkf2 = GroupKFold(n_splits=len(Counter(targ_loc_cat))) ## target word locations
gkf3 = GroupKFold(n_splits=len(sent_len_cat.value_counts())) ## sentence length
```

# Fitted to: BWS 


## fold: target words

```python
X = sentences
y = resp_bws
y_type = 'bws'
```

### including target word

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

## fold: target word locations


### including target word

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

## fold: sent length


### including target word

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

# Fitted to: Semantic distance


## fold: target words

```python
X = sentences
y = resp_brt
y_type = 'brt'
```

### including target word

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

## fold: target word locations


### including target word

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

## fold: sent length


### including target word

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

# Fitted to: Lexical entropy


## fold: target words

```python
X = sentences
y = resp_lex
y_type = 'lex'
```

## fold: sent length


### including target word

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, False, 
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf3.split(df_cloze['sentence'], groups=sent_len_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvSlen", 
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvSlen",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### including target word

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

## fold: target word locations


### including target word

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, False,
                 "./model_weights/model_bert_wttarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_wttarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, True,
                 gkf_split, True,
                 "./model_weights/model_bert_wttarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_wttarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, False,
                 "./model_weights/model_bert_notarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_notarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_bertmod_cv(X, y, False,
                 gkf_split, True,
                 "./model_weights/model_bert_notarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_bert_notarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python

```

# Classification performance 

```python
def roc_cv(cv_true_scores, pred_score_file_loc, score_type, cut, direction, fig, ax, col, ls):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    pred_score_files = sorted(glob.glob(pred_score_file_loc))

    for i in range(len(cv_true_scores)):
        if(direction=="high"):
            fpr, tpr, _ = roc_curve(cv_true_scores[i] > np.quantile(cv_true_scores[i], q=[cut]), np.load(pred_score_files[i]))
        if(direction=="low"):
            fpr, tpr, _ = roc_curve(cv_true_scores[i] < np.quantile(cv_true_scores[i], q=[cut]), 1-np.load(pred_score_files[i]))
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # sns.lineplot(fpr, tpr)
        ax.plot(fpr, tpr, 
                color=col, alpha=0.1,
                # label = 'ROC fold %d (AUC=%0.2f, n=%d)' % (i, roc_auc, fold_set.shape[0])
               )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)    
    ax.plot(mean_fpr, mean_tpr, 
             color=col, alpha=1, linestyle=ls,
             label=r'Mean ROC:'+score_type+' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, 1.96*std_auc),
             lw=2)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC__'+str(cut)+"_"+str(direction)+'_5-fold')
    # ax.legend(loc=4, bbox_to_anchor=(0.5, -0.2))
    ax.legend()
```

```python
sraw_test_cvTwrd = []
sraw_test_cvTloc = []
sraw_test_cvSlen = []

sent_test_cvTwrd = []
sent_test_cvTloc = []
sent_test_cvSlen = []
sent_test_cvTwrd2 = []
sent_test_cvTloc2 = []
sent_test_cvSlen2 = []

resp_bws_cvTwrd = [] 
resp_bws_cvTloc = [] 
resp_bws_cvSlen = [] 
resp_lex_cvTwrd = [] 
resp_lex_cvTloc = [] 
resp_lex_cvSlen = [] 
resp_brt_cvTwrd = [] 
resp_brt_cvTloc = [] 
resp_brt_cvSlen = [] 

for train_idx, test_idx in gkf1.split(df_cloze['sentence'], groups=df_cloze['targ']):
    tt_sents = [sent[test_idx] for sent in sentences]
    sraw_test_cvTwrd.append(tt_sents)
    tt_bws = [resp_bws[i] for i in test_idx]
    tt_brt = [resp_brt[i] for i in test_idx]
    tt_lex = [resp_lex[i] for i in test_idx]
    resp_bws_cvTwrd.append(tt_bws)
    resp_brt_cvTwrd.append(tt_brt)
    resp_lex_cvTwrd.append(tt_lex)
    tt_cvTwrd = convert_text_to_examples(tt_sents[0], tt_sents[1], tt_bws)
    
    sent_test_cvTwrd.append(convert_examples_to_features(tokenizer, tt_cvTwrd, True, MAX_SEQ_LEN))
    sent_test_cvTwrd2.append(convert_examples_to_features(tokenizer, tt_cvTwrd, False, MAX_SEQ_LEN))
    
for train_idx, test_idx in gkf2.split(df_cloze['sentence'], groups=targ_loc_cat):
    tt_sents = [sent[test_idx] for sent in sentences]
    sraw_test_cvTloc.append(tt_sents)
    tt_bws = [resp_bws[i] for i in test_idx]
    tt_brt = [resp_brt[i] for i in test_idx]
    tt_lex = [resp_lex[i] for i in test_idx]
    resp_bws_cvTloc.append(tt_bws)
    resp_brt_cvTloc.append(tt_brt)
    resp_lex_cvTloc.append(tt_lex)
    tt_cvTloc = convert_text_to_examples(tt_sents[0], tt_sents[1], tt_bws)
    
    sent_test_cvTloc.append(convert_examples_to_features(tokenizer, tt_cvTloc, True, MAX_SEQ_LEN))
    sent_test_cvTloc2.append(convert_examples_to_features(tokenizer, tt_cvTloc, False, MAX_SEQ_LEN))
    
for train_idx, test_idx in gkf3.split(df_cloze['sentence'], groups=sent_len_cat):
    tt_sents = [sent[test_idx] for sent in sentences]
    sraw_test_cvSlen.append(tt_sents)
    tt_bws = [resp_bws[i] for i in test_idx]
    tt_brt = [resp_brt[i] for i in test_idx]
    tt_lex = [resp_lex[i] for i in test_idx]
    resp_bws_cvSlen.append(tt_bws)
    resp_brt_cvSlen.append(tt_brt)
    resp_lex_cvSlen.append(tt_lex)
    tt_cvSlen = convert_text_to_examples(tt_sents[0], tt_sents[1], tt_bws)
    
    sent_test_cvSlen.append(convert_examples_to_features(tokenizer, tt_cvSlen, True, MAX_SEQ_LEN))
    sent_test_cvSlen2.append(convert_examples_to_features(tokenizer, tt_cvSlen, False, MAX_SEQ_LEN))
```

## CV over target words vs. target word location

```python
Counter(targ_loc_cat)
```

```python
sent_len_cat.value_counts()
```

### fitted to: BWS score
- target location may bias the prediction results in some cases, but not by much
- sentence length may be the problem
- attention layer improve the performance
    - both with/w.o. target word conditions
- NOT including the target word (synonym of the original) improves the performance
- Attention improves the performance
    - attention captures: the relationship between the target (known or unknown) and context

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[3], '-')

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[3], '-')

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[3], '-')

```

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[3], '-')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_noattn_bws_cvTloc*", "bert_wttarg_noattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[0], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTloc*", "bert_wttarg_wtattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[1], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_noattn_bws_cvTloc*", "bert_notarg_noattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[2], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_wtattn_bws_cvTloc*", "bert_notarg_wtattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[3], '--')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_noattn_bws_cvSlen*", "bert_wttarg_noattn_bws_cvSlen", 0.50, "high", fig, axes[0], tt_col[0], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_bws_cvSlen*", "bert_wttarg_wtattn_bws_cvSlen", 0.50, "high", fig, axes[0], tt_col[1], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_noattn_bws_cvSlen*", "bert_notarg_noattn_bws_cvSlen", 0.50, "high", fig, axes[0], tt_col[2], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_wtattn_bws_cvSlen*", "bert_notarg_wtattn_bws_cvSlen", 0.50, "high", fig, axes[0], tt_col[3], ':')

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[3], '-')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_noattn_bws_cvTloc*", "bert_wttarg_noattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[0], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTloc*", "bert_wttarg_wtattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[1], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_noattn_bws_cvTloc*", "bert_notarg_noattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[2], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_wtattn_bws_cvTloc*", "bert_notarg_wtattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[3], '--')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_noattn_bws_cvSlen*", "bert_wttarg_noattn_bws_cvSlen", 0.25, "high", fig, axes[1], tt_col[0], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_bws_cvSlen*", "bert_wttarg_wtattn_bws_cvSlen", 0.25, "high", fig, axes[1], tt_col[1], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_noattn_bws_cvSlen*", "bert_notarg_noattn_bws_cvSlen", 0.25, "high", fig, axes[1], tt_col[2], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_wtattn_bws_cvSlen*", "bert_notarg_wtattn_bws_cvSlen", 0.25, "high", fig, axes[1], tt_col[3], ':')

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[3], '-')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_noattn_bws_cvTloc*", "bert_wttarg_noattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[0], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTloc*", "bert_wttarg_wtattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[1], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_noattn_bws_cvTloc*", "bert_notarg_noattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[2], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_wtattn_bws_cvTloc*", "bert_notarg_wtattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[3], '--')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_noattn_bws_cvSlen*", "bert_wttarg_noattn_bws_cvSlen", 0.10, "high", fig, axes[2], tt_col[0], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_bws_cvSlen*", "bert_wttarg_wtattn_bws_cvSlen", 0.10, "high", fig, axes[2], tt_col[1], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_noattn_bws_cvSlen*", "bert_notarg_noattn_bws_cvSlen", 0.10, "high", fig, axes[2], tt_col[2], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_wtattn_bws_cvSlen*", "bert_notarg_wtattn_bws_cvSlen", 0.10, "high", fig, axes[2], tt_col[3], ':')
```

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[3], '-')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_noattn_bws_cvTloc*", "bert_wttarg_noattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[0], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTloc*", "bert_wttarg_wtattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[1], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_noattn_bws_cvTloc*", "bert_notarg_noattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[2], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_wtattn_bws_cvTloc*", "bert_notarg_wtattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[3], '--')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_noattn_bws_cvSlen*", "bert_wttarg_noattn_bws_cvSlen", 0.50, "low", fig, axes[0], tt_col[0], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_bws_cvSlen*", "bert_wttarg_wtattn_bws_cvSlen", 0.50, "low", fig, axes[0], tt_col[1], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_noattn_bws_cvSlen*", "bert_notarg_noattn_bws_cvSlen", 0.50, "low", fig, axes[0], tt_col[2], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_wtattn_bws_cvSlen*", "bert_notarg_wtattn_bws_cvSlen", 0.50, "low", fig, axes[0], tt_col[3], ':')

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[3], '-')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_noattn_bws_cvTloc*", "bert_wttarg_noattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[0], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTloc*", "bert_wttarg_wtattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[1], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_noattn_bws_cvTloc*", "bert_notarg_noattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[2], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_wtattn_bws_cvTloc*", "bert_notarg_wtattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[3], '--')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_noattn_bws_cvSlen*", "bert_wttarg_noattn_bws_cvSlen", 0.25, "low", fig, axes[1], tt_col[0], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_bws_cvSlen*", "bert_wttarg_wtattn_bws_cvSlen", 0.25, "low", fig, axes[1], tt_col[1], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_noattn_bws_cvSlen*", "bert_notarg_noattn_bws_cvSlen", 0.25, "low", fig, axes[1], tt_col[2], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_wtattn_bws_cvSlen*", "bert_notarg_wtattn_bws_cvSlen", 0.25, "low", fig, axes[1], tt_col[3], ':')

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_bws_cvTwrd*", "bert_wttarg_noattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[0], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTwrd*", "bert_wttarg_wtattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[1], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*", "bert_notarg_noattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[2], '-')
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*", "bert_notarg_wtattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[3], '-')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_noattn_bws_cvTloc*", "bert_wttarg_noattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[0], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_bws_cvTloc*", "bert_wttarg_wtattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[1], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_noattn_bws_cvTloc*", "bert_notarg_noattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[2], '--')
roc_cv(resp_bws_cvTloc, "./model_predict/preds_bert_notarg_wtattn_bws_cvTloc*", "bert_notarg_wtattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[3], '--')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_noattn_bws_cvSlen*", "bert_wttarg_noattn_bws_cvSlen", 0.10, "low", fig, axes[2], tt_col[0], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_bws_cvSlen*", "bert_wttarg_wtattn_bws_cvSlen", 0.10, "low", fig, axes[2], tt_col[1], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_noattn_bws_cvSlen*", "bert_notarg_noattn_bws_cvSlen", 0.10, "low", fig, axes[2], tt_col[2], ':')
roc_cv(resp_bws_cvSlen, "./model_predict/preds_bert_notarg_wtattn_bws_cvSlen*", "bert_notarg_wtattn_bws_cvSlen", 0.10, "low", fig, axes[2], tt_col[3], ':')
```

### fitted to: Semantic distance
- Semantic distance score vs. BWS
    - cloze response based score vs. ratings on cloze sentences
    - cloze response require additional cognitive process
        - which may introduce additional human bias in reaponse generation/recall of cloze responses
- target location may bias (worse) the prediction results in some cases, but not by much
- NOT including the target word improves the performance IF the model does not include the attention layer
- attention layer increase the performance if the target word is known
    - but it harms the performance if the target word is unknown
    - humans: recall the cloze response based on context + specific semantic ground?

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_brt_cvTwrd*", "bert_wttarg_noattn_brt_cvTwrd", 0.50, "high", fig, axes[0], tt_col[0], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTwrd*", "bert_wttarg_wtattn_brt_cvTwrd", 0.50, "high", fig, axes[0], tt_col[1], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_noattn_brt_cvTwrd*", "bert_notarg_noattn_brt_cvTwrd", 0.50, "high", fig, axes[0], tt_col[2], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_brt_cvTwrd*", "bert_notarg_wtattn_brt_cvTwrd", 0.50, "high", fig, axes[0], tt_col[3], '-')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_noattn_brt_cvTloc*", "bert_wttarg_noattn_brt_cvTloc", 0.50, "high", fig, axes[0], tt_col[0], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTloc*", "bert_wttarg_wtattn_brt_cvTloc", 0.50, "high", fig, axes[0], tt_col[1], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_noattn_brt_cvTloc*", "bert_notarg_noattn_brt_cvTloc", 0.50, "high", fig, axes[0], tt_col[2], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_wtattn_brt_cvTloc*", "bert_notarg_wtattn_brt_cvTloc", 0.50, "high", fig, axes[0], tt_col[3], '--')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_noattn_brt_cvSlen*", "bert_wttarg_noattn_brt_cvSlen", 0.50, "high", fig, axes[0], tt_col[0], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_brt_cvSlen*", "bert_wttarg_wtattn_brt_cvSlen", 0.50, "high", fig, axes[0], tt_col[1], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_noattn_brt_cvSlen*", "bert_notarg_noattn_brt_cvSlen", 0.50, "high", fig, axes[0], tt_col[2], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_wtattn_brt_cvSlen*", "bert_notarg_wtattn_brt_cvSlen", 0.50, "high", fig, axes[0], tt_col[3], ':')


roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_brt_cvTwrd*", "bert_wttarg_noattn_brt_cvTwrd", 0.25, "high", fig, axes[1], tt_col[0], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTwrd*", "bert_wttarg_wtattn_brt_cvTwrd", 0.25, "high", fig, axes[1], tt_col[1], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_noattn_brt_cvTwrd*", "bert_notarg_noattn_brt_cvTwrd", 0.25, "high", fig, axes[1], tt_col[2], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_brt_cvTwrd*", "bert_notarg_wtattn_brt_cvTwrd", 0.25, "high", fig, axes[1], tt_col[3], '-')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_noattn_brt_cvTloc*", "bert_wttarg_noattn_brt_cvTloc", 0.25, "high", fig, axes[1], tt_col[0], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTloc*", "bert_wttarg_wtattn_brt_cvTloc", 0.25, "high", fig, axes[1], tt_col[1], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_noattn_brt_cvTloc*", "bert_notarg_noattn_brt_cvTloc", 0.25, "high", fig, axes[1], tt_col[2], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_wtattn_brt_cvTloc*", "bert_notarg_wtattn_brt_cvTloc", 0.25, "high", fig, axes[1], tt_col[3], '--')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_noattn_brt_cvSlen*", "bert_wttarg_noattn_brt_cvSlen", 0.25, "high", fig, axes[1], tt_col[0], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_brt_cvSlen*", "bert_wttarg_wtattn_brt_cvSlen", 0.25, "high", fig, axes[1], tt_col[1], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_noattn_brt_cvSlen*", "bert_notarg_noattn_brt_cvSlen", 0.25, "high", fig, axes[1], tt_col[2], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_wtattn_brt_cvSlen*", "bert_notarg_wtattn_brt_cvSlen", 0.25, "high", fig, axes[1], tt_col[3], ':')

roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_brt_cvTwrd*", "bert_wttarg_noattn_brt_cvTwrd", 0.10, "high", fig, axes[2], tt_col[0], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTwrd*", "bert_wttarg_wtattn_brt_cvTwrd", 0.10, "high", fig, axes[2], tt_col[1], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_noattn_brt_cvTwrd*", "bert_notarg_noattn_brt_cvTwrd", 0.10, "high", fig, axes[2], tt_col[2], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_brt_cvTwrd*", "bert_notarg_wtattn_brt_cvTwrd", 0.10, "high", fig, axes[2], tt_col[3], '-')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_noattn_brt_cvTloc*", "bert_wttarg_noattn_brt_cvTloc", 0.10, "high", fig, axes[2], tt_col[0], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTloc*", "bert_wttarg_wtattn_brt_cvTloc", 0.10, "high", fig, axes[2], tt_col[1], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_noattn_brt_cvTloc*", "bert_notarg_noattn_brt_cvTloc", 0.10, "high", fig, axes[2], tt_col[2], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_wtattn_brt_cvTloc*", "bert_notarg_wtattn_brt_cvTloc", 0.10, "high", fig, axes[2], tt_col[3], '--')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_noattn_brt_cvSlen*", "bert_wttarg_noattn_brt_cvSlen", 0.10, "high", fig, axes[2], tt_col[0], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_brt_cvSlen*", "bert_wttarg_wtattn_brt_cvSlen", 0.10, "high", fig, axes[2], tt_col[1], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_noattn_brt_cvSlen*", "bert_notarg_noattn_brt_cvSlen", 0.10, "high", fig, axes[2], tt_col[2], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_wtattn_brt_cvSlen*", "bert_notarg_wtattn_brt_cvSlen", 0.10, "high", fig, axes[2], tt_col[3], ':')


```

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_brt_cvTwrd*", "bert_wttarg_noattn_brt_cvTwrd", 0.50, "low", fig, axes[0], tt_col[0], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTwrd*", "bert_wttarg_wtattn_brt_cvTwrd", 0.50, "low", fig, axes[0], tt_col[1], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_noattn_brt_cvTwrd*", "bert_notarg_noattn_brt_cvTwrd", 0.50, "low", fig, axes[0], tt_col[2], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_brt_cvTwrd*", "bert_notarg_wtattn_brt_cvTwrd", 0.50, "low", fig, axes[0], tt_col[3], '-')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_noattn_brt_cvTloc*", "bert_wttarg_noattn_brt_cvTloc", 0.50, "low", fig, axes[0], tt_col[0], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTloc*", "bert_wttarg_wtattn_brt_cvTloc", 0.50, "low", fig, axes[0], tt_col[1], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_noattn_brt_cvTloc*", "bert_notarg_noattn_brt_cvTloc", 0.50, "low", fig, axes[0], tt_col[2], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_wtattn_brt_cvTloc*", "bert_notarg_wtattn_brt_cvTloc", 0.50, "low", fig, axes[0], tt_col[3], '--')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_noattn_brt_cvSlen*", "bert_wttarg_noattn_brt_cvSlen", 0.50, "low", fig, axes[0], tt_col[0], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_brt_cvSlen*", "bert_wttarg_wtattn_brt_cvSlen", 0.50, "low", fig, axes[0], tt_col[1], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_noattn_brt_cvSlen*", "bert_notarg_noattn_brt_cvSlen", 0.50, "low", fig, axes[0], tt_col[2], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_wtattn_brt_cvSlen*", "bert_notarg_wtattn_brt_cvSlen", 0.50, "low", fig, axes[0], tt_col[3], ':')


roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_brt_cvTwrd*", "bert_wttarg_noattn_brt_cvTwrd", 0.25, "low", fig, axes[1], tt_col[0], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTwrd*", "bert_wttarg_wtattn_brt_cvTwrd", 0.25, "low", fig, axes[1], tt_col[1], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_noattn_brt_cvTwrd*", "bert_notarg_noattn_brt_cvTwrd", 0.25, "low", fig, axes[1], tt_col[2], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_brt_cvTwrd*", "bert_notarg_wtattn_brt_cvTwrd", 0.25, "low", fig, axes[1], tt_col[3], '-')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_noattn_brt_cvTloc*", "bert_wttarg_noattn_brt_cvTloc", 0.25, "low", fig, axes[1], tt_col[0], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTloc*", "bert_wttarg_wtattn_brt_cvTloc", 0.25, "low", fig, axes[1], tt_col[1], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_noattn_brt_cvTloc*", "bert_notarg_noattn_brt_cvTloc", 0.25, "low", fig, axes[1], tt_col[2], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_wtattn_brt_cvTloc*", "bert_notarg_wtattn_brt_cvTloc", 0.25, "low", fig, axes[1], tt_col[3], '--')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_noattn_brt_cvSlen*", "bert_wttarg_noattn_brt_cvSlen", 0.25, "low", fig, axes[1], tt_col[0], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_brt_cvSlen*", "bert_wttarg_wtattn_brt_cvSlen", 0.25, "low", fig, axes[1], tt_col[1], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_noattn_brt_cvSlen*", "bert_notarg_noattn_brt_cvSlen", 0.25, "low", fig, axes[1], tt_col[2], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_wtattn_brt_cvSlen*", "bert_notarg_wtattn_brt_cvSlen", 0.25, "low", fig, axes[1], tt_col[3], ':')

roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_brt_cvTwrd*", "bert_wttarg_noattn_brt_cvTwrd", 0.10, "low", fig, axes[2], tt_col[0], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTwrd*", "bert_wttarg_wtattn_brt_cvTwrd", 0.10, "low", fig, axes[2], tt_col[1], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_noattn_brt_cvTwrd*", "bert_notarg_noattn_brt_cvTwrd", 0.10, "low", fig, axes[2], tt_col[2], '-')
roc_cv(resp_brt_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_brt_cvTwrd*", "bert_notarg_wtattn_brt_cvTwrd", 0.10, "low", fig, axes[2], tt_col[3], '-')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_noattn_brt_cvTloc*", "bert_wttarg_noattn_brt_cvTloc", 0.10, "low", fig, axes[2], tt_col[0], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_brt_cvTloc*", "bert_wttarg_wtattn_brt_cvTloc", 0.10, "low", fig, axes[2], tt_col[1], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_noattn_brt_cvTloc*", "bert_notarg_noattn_brt_cvTloc", 0.10, "low", fig, axes[2], tt_col[2], '--')
roc_cv(resp_brt_cvTloc, "./model_predict/preds_bert_notarg_wtattn_brt_cvTloc*", "bert_notarg_wtattn_brt_cvTloc", 0.10, "low", fig, axes[2], tt_col[3], '--')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_noattn_brt_cvSlen*", "bert_wttarg_noattn_brt_cvSlen", 0.10, "low", fig, axes[2], tt_col[0], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_brt_cvSlen*", "bert_wttarg_wtattn_brt_cvSlen", 0.10, "low", fig, axes[2], tt_col[1], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_noattn_brt_cvSlen*", "bert_notarg_noattn_brt_cvSlen", 0.10, "low", fig, axes[2], tt_col[2], ':')
roc_cv(resp_brt_cvSlen, "./model_predict/preds_bert_notarg_wtattn_brt_cvSlen*", "bert_notarg_wtattn_brt_cvSlen", 0.10, "low", fig, axes[2], tt_col[3], ':')


```

### fitted to: lexical entropy
- target location may bias (improve) the prediction results in some cases, but not by much
- Similar to semantic distance score results: 
    - NOT including the target word improves the performance IF the model does not include the attention layer
    - attention layer increase the performance if the target word is known
        - but nearly no differences if the target word is unknown


```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_lex_cvTwrd*", "bert_wttarg_noattn_lex_cvTwrd", 0.50, "high", fig, axes[0], tt_col[0], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTwrd*", "bert_wttarg_wtattn_lex_cvTwrd", 0.50, "high", fig, axes[0], tt_col[1], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_noattn_lex_cvTwrd*", "bert_notarg_noattn_lex_cvTwrd", 0.50, "high", fig, axes[0], tt_col[2], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_lex_cvTwrd*", "bert_notarg_wtattn_lex_cvTwrd", 0.50, "high", fig, axes[0], tt_col[3], '-')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_noattn_lex_cvTloc*", "bert_wttarg_noattn_lex_cvTloc", 0.50, "high", fig, axes[0], tt_col[0], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTloc*", "bert_wttarg_wtattn_lex_cvTloc", 0.50, "high", fig, axes[0], tt_col[1], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_noattn_lex_cvTloc*", "bert_notarg_noattn_lex_cvTloc", 0.50, "high", fig, axes[0], tt_col[2], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_wtattn_lex_cvTloc*", "bert_notarg_wtattn_lex_cvTloc", 0.50, "high", fig, axes[0], tt_col[3], '--')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_noattn_lex_cvSlen*", "bert_wttarg_noattn_lex_cvSlen", 0.50, "high", fig, axes[0], tt_col[0], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_lex_cvSlen*", "bert_wttarg_wtattn_lex_cvSlen", 0.50, "high", fig, axes[0], tt_col[1], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_noattn_lex_cvSlen*", "bert_notarg_noattn_lex_cvSlen", 0.50, "high", fig, axes[0], tt_col[2], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_wtattn_lex_cvSlen*", "bert_notarg_wtattn_lex_cvSlen", 0.50, "high", fig, axes[0], tt_col[3], ':')

roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_lex_cvTwrd*", "bert_wttarg_noattn_lex_cvTwrd", 0.25, "high", fig, axes[1], tt_col[0], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTwrd*", "bert_wttarg_wtattn_lex_cvTwrd", 0.25, "high", fig, axes[1], tt_col[1], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_noattn_lex_cvTwrd*", "bert_notarg_noattn_lex_cvTwrd", 0.25, "high", fig, axes[1], tt_col[2], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_lex_cvTwrd*", "bert_notarg_wtattn_lex_cvTwrd", 0.25, "high", fig, axes[1], tt_col[3], '-')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_noattn_lex_cvTloc*", "bert_wttarg_noattn_lex_cvTloc", 0.25, "high", fig, axes[1], tt_col[0], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTloc*", "bert_wttarg_wtattn_lex_cvTloc", 0.25, "high", fig, axes[1], tt_col[1], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_noattn_lex_cvTloc*", "bert_notarg_noattn_lex_cvTloc", 0.25, "high", fig, axes[1], tt_col[2], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_wtattn_lex_cvTloc*", "bert_notarg_wtattn_lex_cvTloc", 0.25, "high", fig, axes[1], tt_col[3], '--')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_noattn_lex_cvSlen*", "bert_wttarg_noattn_lex_cvSlen", 0.25, "high", fig, axes[1], tt_col[0], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_lex_cvSlen*", "bert_wttarg_wtattn_lex_cvSlen", 0.25, "high", fig, axes[1], tt_col[1], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_noattn_lex_cvSlen*", "bert_notarg_noattn_lex_cvSlen", 0.25, "high", fig, axes[1], tt_col[2], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_wtattn_lex_cvSlen*", "bert_notarg_wtattn_lex_cvSlen", 0.25, "high", fig, axes[1], tt_col[3], ':')

roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_lex_cvTwrd*", "bert_wttarg_noattn_lex_cvTwrd", 0.10, "high", fig, axes[2], tt_col[0], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTwrd*", "bert_wttarg_wtattn_lex_cvTwrd", 0.10, "high", fig, axes[2], tt_col[1], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_noattn_lex_cvTwrd*", "bert_notarg_noattn_lex_cvTwrd", 0.10, "high", fig, axes[2], tt_col[2], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_lex_cvTwrd*", "bert_notarg_wtattn_lex_cvTwrd", 0.10, "high", fig, axes[2], tt_col[3], '-')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_noattn_lex_cvTloc*", "bert_wttarg_noattn_lex_cvTloc", 0.10, "high", fig, axes[2], tt_col[0], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTloc*", "bert_wttarg_wtattn_lex_cvTloc", 0.10, "high", fig, axes[2], tt_col[1], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_noattn_lex_cvTloc*", "bert_notarg_noattn_lex_cvTloc", 0.10, "high", fig, axes[2], tt_col[2], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_wtattn_lex_cvTloc*", "bert_notarg_wtattn_lex_cvTloc", 0.10, "high", fig, axes[2], tt_col[3], '--')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_noattn_lex_cvSlen*", "bert_wttarg_noattn_lex_cvSlen", 0.10, "high", fig, axes[2], tt_col[0], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_lex_cvSlen*", "bert_wttarg_wtattn_lex_cvSlen", 0.10, "high", fig, axes[2], tt_col[1], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_noattn_lex_cvSlen*", "bert_notarg_noattn_lex_cvSlen", 0.10, "high", fig, axes[2], tt_col[2], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_wtattn_lex_cvSlen*", "bert_notarg_wtattn_lex_cvSlen", 0.10, "high", fig, axes[2], tt_col[3], ':')

```

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_lex_cvTwrd*", "bert_wttarg_noattn_lex_cvTwrd", 0.50, "low", fig, axes[0], tt_col[0], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTwrd*", "bert_wttarg_wtattn_lex_cvTwrd", 0.50, "low", fig, axes[0], tt_col[1], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_noattn_lex_cvTwrd*", "bert_notarg_noattn_lex_cvTwrd", 0.50, "low", fig, axes[0], tt_col[2], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_lex_cvTwrd*", "bert_notarg_wtattn_lex_cvTwrd", 0.50, "low", fig, axes[0], tt_col[3], '-')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_noattn_lex_cvTloc*", "bert_wttarg_noattn_lex_cvTloc", 0.50, "low", fig, axes[0], tt_col[0], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTloc*", "bert_wttarg_wtattn_lex_cvTloc", 0.50, "low", fig, axes[0], tt_col[1], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_noattn_lex_cvTloc*", "bert_notarg_noattn_lex_cvTloc", 0.50, "low", fig, axes[0], tt_col[2], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_wtattn_lex_cvTloc*", "bert_notarg_wtattn_lex_cvTloc", 0.50, "low", fig, axes[0], tt_col[3], '--')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_noattn_lex_cvSlen*", "bert_wttarg_noattn_lex_cvSlen", 0.50, "low", fig, axes[0], tt_col[0], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_lex_cvSlen*", "bert_wttarg_wtattn_lex_cvSlen", 0.50, "low", fig, axes[0], tt_col[1], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_noattn_lex_cvSlen*", "bert_notarg_noattn_lex_cvSlen", 0.50, "low", fig, axes[0], tt_col[2], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_wtattn_lex_cvSlen*", "bert_notarg_wtattn_lex_cvSlen", 0.50, "low", fig, axes[0], tt_col[3], ':')

roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_lex_cvTwrd*", "bert_wttarg_noattn_lex_cvTwrd", 0.25, "low", fig, axes[1], tt_col[0], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTwrd*", "bert_wttarg_wtattn_lex_cvTwrd", 0.25, "low", fig, axes[1], tt_col[1], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_noattn_lex_cvTwrd*", "bert_notarg_noattn_lex_cvTwrd", 0.25, "low", fig, axes[1], tt_col[2], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_lex_cvTwrd*", "bert_notarg_wtattn_lex_cvTwrd", 0.25, "low", fig, axes[1], tt_col[3], '-')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_noattn_lex_cvTloc*", "bert_wttarg_noattn_lex_cvTloc", 0.25, "low", fig, axes[1], tt_col[0], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTloc*", "bert_wttarg_wtattn_lex_cvTloc", 0.25, "low", fig, axes[1], tt_col[1], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_noattn_lex_cvTloc*", "bert_notarg_noattn_lex_cvTloc", 0.25, "low", fig, axes[1], tt_col[2], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_wtattn_lex_cvTloc*", "bert_notarg_wtattn_lex_cvTloc", 0.25, "low", fig, axes[1], tt_col[3], '--')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_noattn_lex_cvSlen*", "bert_wttarg_noattn_lex_cvSlen", 0.25, "low", fig, axes[1], tt_col[0], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_lex_cvSlen*", "bert_wttarg_wtattn_lex_cvSlen", 0.25, "low", fig, axes[1], tt_col[1], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_noattn_lex_cvSlen*", "bert_notarg_noattn_lex_cvSlen", 0.25, "low", fig, axes[1], tt_col[2], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_wtattn_lex_cvSlen*", "bert_notarg_wtattn_lex_cvSlen", 0.25, "low", fig, axes[1], tt_col[3], ':')

roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_noattn_lex_cvTwrd*", "bert_wttarg_noattn_lex_cvTwrd", 0.10, "low", fig, axes[2], tt_col[0], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTwrd*", "bert_wttarg_wtattn_lex_cvTwrd", 0.10, "low", fig, axes[2], tt_col[1], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_noattn_lex_cvTwrd*", "bert_notarg_noattn_lex_cvTwrd", 0.10, "low", fig, axes[2], tt_col[2], '-')
roc_cv(resp_lex_cvTwrd, "./model_predict/preds_bert_notarg_wtattn_lex_cvTwrd*", "bert_notarg_wtattn_lex_cvTwrd", 0.10, "low", fig, axes[2], tt_col[3], '-')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_noattn_lex_cvTloc*", "bert_wttarg_noattn_lex_cvTloc", 0.10, "low", fig, axes[2], tt_col[0], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_wttarg_wtattn_lex_cvTloc*", "bert_wttarg_wtattn_lex_cvTloc", 0.10, "low", fig, axes[2], tt_col[1], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_noattn_lex_cvTloc*", "bert_notarg_noattn_lex_cvTloc", 0.10, "low", fig, axes[2], tt_col[2], '--')
roc_cv(resp_lex_cvTloc, "./model_predict/preds_bert_notarg_wtattn_lex_cvTloc*", "bert_notarg_wtattn_lex_cvTloc", 0.10, "low", fig, axes[2], tt_col[3], '--')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_noattn_lex_cvSlen*", "bert_wttarg_noattn_lex_cvSlen", 0.10, "low", fig, axes[2], tt_col[0], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_wttarg_wtattn_lex_cvSlen*", "bert_wttarg_wtattn_lex_cvSlen", 0.10, "low", fig, axes[2], tt_col[1], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_noattn_lex_cvSlen*", "bert_notarg_noattn_lex_cvSlen", 0.10, "low", fig, axes[2], tt_col[2], ':')
roc_cv(resp_lex_cvSlen, "./model_predict/preds_bert_notarg_wtattn_lex_cvSlen*", "bert_notarg_wtattn_lex_cvSlen", 0.10, "low", fig, axes[2], tt_col[3], ':')

```

```python

```

# Prediction results 

```python
sent_len_cat.value_counts()
```

```python
[len(x) for x in resp_bws_cvSlen]
```

## score predictions 

```python

```

```python
# https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
from scipy import stats
def spearmanr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.spearmanr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(len(x)-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return {"rho":r, "p-value":p, "ci_low":lo, "ci_high":hi}
```

```python
K.clear_session()
sess = tf.Session()

tt_mod_s = build_model_bert(MAX_SEQ_LEN, True)
tt_mod_m = build_model_bert(MAX_SEQ_LEN, True)
tt_mod_l = build_model_bert(MAX_SEQ_LEN, True)

initialize_vars(sess)
tt_mod_s.load_weights("model_weights/model_bert_notarg_wtattn_bws_cvTwrd0.h5")
tt_mod_m.load_weights("model_weights/model_bert_notarg_wtattn_bws_cvTwrd2.h5")
tt_mod_l.load_weights("model_weights/model_bert_notarg_wtattn_bws_cvTwrd3.h5")
```

```python
# tt_sents = []
# for i, sents in enumerate(sent_test_cvSlen):
#     tt_examples = convert_text_to_examples(sents[0], sents[1], resp_bws_cvSlen[i])
#     tt_sents.append(convert_examples_to_features(tokenizer, tt_examples, False, MAX_SEQ_LEN)[:4])

# attn_bws_pred_test = np.reshape(model.predict(sent_test_cvTwrd[0], batch_size=128), -1)
tt_pred_s = np.reshape(tt_mod_s.predict(sent_test_cvSlen[0], batch_size=128), -1)
tt_pred_m = np.reshape(tt_mod_m.predict(sent_test_cvSlen[2], batch_size=128), -1)
tt_pred_l = np.reshape(tt_mod_l.predict(sent_test_cvSlen[3], batch_size=128), -1)
```

```python
_preds = tt_pred_s
_true = resp_bws_cvSlen[0]
plt.figure(figsize=(6,6))
sns.scatterplot(_true, _preds, alpha=0.3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("BWS: Observed")
plt.ylabel("BWS: notarg_wtattn_Predicted")
# print(spearmanr(_true, _preds))
print(spearmanr_ci(_true, _preds))
print(mean_absolute_error(_true, _preds))
```

```python
_preds = tt_pred_m
_true = resp_bws_cvSlen[2]
plt.figure(figsize=(6,6))
sns.scatterplot(_true, _preds, alpha=0.3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("BWS: Observed")
plt.ylabel("BWS: notarg_wtattn_Predicted")
# print(spearmanr(_true, _preds))
print(spearmanr_ci(_true, _preds))
print(mean_absolute_error(_true, _preds))
```

```python
_preds = tt_pred_l
_true = resp_bws_cvSlen[3]
plt.figure(figsize=(6,6))
sns.scatterplot(_true, _preds, alpha=0.3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("BWS: Observed")
plt.ylabel("BWS: notarg_wtattn_Predicted")
# print(spearmanr(_true, _preds))
print(spearmanr_ci(_true, _preds))
print(mean_absolute_error(_true, _preds))
```

```python

```

```python

```

```python
K.clear_session()
sess = tf.Session()

model_wttarg_noattn = build_model_bert(MAX_SEQ_LEN, False)
model_wttarg_wtattn = build_model_bert(MAX_SEQ_LEN, True)
model_notarg_noattn = build_model_bert(MAX_SEQ_LEN, False)
model_notarg_wtattn = build_model_bert(MAX_SEQ_LEN, True)
initialize_vars(sess)

model_wttarg_noattn.load_weights("model_weights/model_bert_wttarg_noattn_bws_cvTwrd0.h5")
model_wttarg_wtattn.load_weights("model_weights/model_bert_wttarg_wtattn_bws_cvTwrd0.h5")
model_notarg_noattn.load_weights("model_weights/model_bert_notarg_noattn_bws_cvTwrd0.h5")
model_notarg_wtattn.load_weights("model_weights/model_bert_notarg_wtattn_bws_cvTwrd0.h5")
```

```python
# attn_bws_pred_test = np.reshape(model.predict(sent_test_cvTwrd[0], batch_size=128), -1)
wttarg_noattn_pred_test = np.reshape(model_wttarg_noattn.predict(sent_test_cvTwrd[0], batch_size=128), -1)
wttarg_wtattn_pred_test = np.reshape(model_wttarg_wtattn.predict(sent_test_cvTwrd[0], batch_size=128), -1)
notarg_noattn_pred_test = np.reshape(model_notarg_noattn.predict(sent_test_cvTwrd2[0], batch_size=128), -1)
notarg_wtattn_pred_test = np.reshape(model_notarg_wtattn.predict(sent_test_cvTwrd2[0], batch_size=128), -1)
```

### for fold 0

```python
_preds = notarg_wtattn_pred_test
_true = resp_bws_cvTwrd[0]
plt.figure(figsize=(6,6))
sns.scatterplot(_true, _preds, alpha=0.3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("BWS: Observed")
plt.ylabel("BWS: notarg_wtattn_Predicted")
# print(spearmanr(_true, _preds))
print(spearmanr_ci(_true, _preds))
print(mean_absolute_error(_true, _preds))
```

```python
_preds = notarg_noattn_pred_test
_true = resp_bws_cvTwrd[0]
plt.figure(figsize=(6,6))
sns.scatterplot(_true, _preds, alpha=0.3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("BWS: Observed")
plt.ylabel("BWS: notarg_noattn_Predicted")
print(spearmanr_ci(_true, _preds))
print(mean_absolute_error(_true, _preds))
```

```python
sns.scatterplot(notarg_noattn_pred_test, notarg_wtattn_pred_test, alpha=0.3)
```

### consolidating predictions from all cv folds

```python
tt_files = sorted(glob.glob("./model_predict/preds_bert_notarg_wtattn_bws_cvTwrd*"))
tt_preds = [np.load(f) for f in tt_files]
tt_preds = [x for xx in tt_preds for x in xx]
tt_obs = sum(resp_bws_cvTwrd, [])

plt.figure(figsize=(6,6))
sns.scatterplot(tt_obs, tt_preds, alpha=0.3)
print(spearmanr_ci(tt_obs, tt_preds))
print("MAE: ", round(np.abs(np.array(tt_obs) - np.array(tt_preds)).mean(), 3),
      "(±", round(np.abs(np.array(tt_obs) - np.array(tt_preds)).std()*1.96, 3), ")")
```

```python
np.abs(np.array(tt_obs) - np.array(tt_preds)).std()
```

```python
tt_files = sorted(glob.glob("./model_predict/preds_bert_notarg_noattn_bws_cvTwrd*"))
tt_preds = [np.load(f) for f in tt_files]
tt_preds = [x for xx in tt_preds for x in xx]
tt_obs = sum(resp_bws_cvTwrd, [])

plt.figure(figsize=(6,6))
sns.scatterplot(tt_obs, tt_preds, alpha=0.3)
print(spearmanr_ci(tt_obs, tt_preds))
print("MAE: ", round(np.abs(np.array(tt_obs) - np.array(tt_preds)).mean(), 3),
      "(±", round(np.abs(np.array(tt_obs) - np.array(tt_preds)).std()*1.96, 3), ")")
```

```python

```

## attention interpretations

```python colab={} colab_type="code" id="1zpBhTwiyaT_"
model_notarg_sfmxres = Model(model_notarg_wtattn.inputs, model_notarg_wtattn.get_layer('attention_layer').output[1])
model_wttarg_sfmxres = Model(model_wttarg_wtattn.inputs, model_wttarg_wtattn.get_layer('attention_layer').output[1])
# initialize_vars(sess)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 2458085, "status": "ok", "timestamp": 1569436612264, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="r_nzXPiQbu61" outputId="c2c93a03-1d93-4d5b-ee63-64b9003a4b5a"
hi_test_idx15 = np.prod([(resp_bws_cvTwrd[0] > np.quantile(resp_bws_cvTwrd[0], q=[0.90])),
                         (resp_brt_cvTwrd[0] > np.quantile(resp_brt_cvTwrd[0], q=[0.90]))], axis=0)
lo_test_idx15 = np.prod([(resp_bws_cvTwrd[0] < np.quantile(resp_bws_cvTwrd[0], q=[0.13])),
                         (resp_brt_cvTwrd[0] < np.quantile(resp_brt_cvTwrd[0], q=[0.13]))], axis=0)
hi_test_idx15 = np.where(hi_test_idx15)[0]
lo_test_idx15 = np.where(lo_test_idx15)[0]
(hi_test_idx15, lo_test_idx15)
```

```python colab={} colab_type="code" id="XllPYkfLyfAZ"
tt_notarg_attn = model_notarg_sfmxres.predict([sent for sent in sent_test_cvTwrd2[0]])[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict([sent for sent in sent_test_cvTwrd[0]])[:,0,:]
```

#### high informative sentences

```python
def min_max_sc(_x): 
    x = np.array(_x)
    return (x-x.min())/(x.max()-x.min())
```

```python
np.not_equal(sent_test_cvTwrd2[0][0], 0)
```

```python
def draw_attn_tok(_tokenizer, attn_weights, sent_input, true_score, pred_score, _label, ax, _align_edge=None):
    attn_score = attn_weights * np.not_equal(sent_input[0], 0) #1 #sent_input[1]
    attn_score = min_max_sc(attn_score)
    
    if(_align_edge is None):
        ax.bar(x=_tokenizer.convert_ids_to_tokens(sent_input[0]), height=attn_score, width=-0.25, label=_label)
    else:
        ax.bar(x=_tokenizer.convert_ids_to_tokens(sent_input[0]), height=attn_score, width=0.15, label=_label, align='edge')
    ax.set_title(_label + " BWS score: "+str(round(true_score, 3))+
                 ", pred score: "+str(round(pred_score, 3)))    
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
```

```python
fig, axes = plt.subplots(len(hi_test_idx15), 2, figsize=(16, len(hi_test_idx15)*3))
fig.subplots_adjust(hspace=2)

for i,j in enumerate(hi_test_idx15):
    draw_attn_tok(tokenizer, tt_notarg_attn[j], [sent[j] for sent in sent_test_cvTwrd2[0]], resp_bws_cvTwrd[0][j], notarg_wtattn_pred_test[j], "w.o. targ", axes[i][0])
    draw_attn_tok(tokenizer, tt_wttarg_attn[j], [sent[j] for sent in sent_test_cvTwrd[0]], resp_bws_cvTwrd[0][j], wttarg_wtattn_pred_test[j], "with targ", axes[i][1])
```

#### low informative sentences

```python
fig, axes = plt.subplots(len(lo_test_idx15), 2, figsize=(16, len(lo_test_idx15)*3))
fig.subplots_adjust(hspace=2)

for i,j in enumerate(lo_test_idx15):
    draw_attn_tok(tokenizer, tt_notarg_attn[j], [sent[j] for sent in sent_test_cvTwrd2[0]], resp_bws_cvTwrd[0][j], notarg_wtattn_pred_test[j], "w.o. targ", axes[i][0])
    draw_attn_tok(tokenizer, tt_wttarg_attn[j], [sent[j] for sent in sent_test_cvTwrd[0]], resp_bws_cvTwrd[0][j], wttarg_wtattn_pred_test[j], "with targ", axes[i][1])
```

```python

```

<!-- #region {"colab_type": "text", "id": "8XyI2OhJn57I"} -->
# Anecdotal examples - High informative sentences
- context cue categories (based on DSCoVAR_TR2016-005_AnnotatingContextCues.docx)
    - Synonymous
    - Antonymous
    - Causal
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 762} colab_type="code" executionInfo={"elapsed": 2498752, "status": "ok", "timestamp": 1569436652981, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="uzzLbG6MtT1H" outputId="d2ecd183-5be6-470e-abca-e672f3a52107"
df_ex_hinfo = pd.read_csv("./dataset/hiinfo_examples.csv")
df_ex_hinfo
```

```python colab={} colab_type="code" id="UpjzTe0y3cYl"
sentences_high = proc_sentences(df_ex_hinfo, "Example context", "syn1")
sentences_high2 = proc_sentences(df_ex_hinfo, "Example context", None)
```

```python colab={} colab_type="code" id="pt_hk9VrxXjI"
sentences_high_syn = [xx[df_ex_hinfo["Cue "]=="Synonym"] for xx in sentences_high]
sentences_high_ant = [xx[df_ex_hinfo["Cue "]=="Antonym"] for xx in sentences_high]
sentences_high_cau = [xx[df_ex_hinfo["Cue "]=="Causal"]  for xx in sentences_high]
sentences_high2_syn = [xx[df_ex_hinfo["Cue "]=="Synonym"] for xx in sentences_high2]
sentences_high2_ant = [xx[df_ex_hinfo["Cue "]=="Antonym"] for xx in sentences_high2]
sentences_high2_cau = [xx[df_ex_hinfo["Cue "]=="Causal"]  for xx in sentences_high2]
```

<!-- #region {"colab_type": "text", "id": "Ji05EQlln8zt"} -->
## synonym
- w.o. target word: more emphasize on 'skeptical', which is a synonym cue
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 902} colab_type="code" executionInfo={"elapsed": 11361, "status": "ok", "timestamp": 1569439342415, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="rJ9449R6yaPe" outputId="d2412c34-73a5-49e5-e92a-01bf091ee85e"
tt_sent = sentences_high_syn
tt_test  = convert_text_to_examples(tt_sent[0], tt_sent[1], np.zeros(len(tt_sent[0])))
sent_test  = convert_examples_to_features(tokenizer, tt_test, True, MAX_SEQ_LEN)
sent_test2 = convert_examples_to_features(tokenizer, tt_test, False, MAX_SEQ_LEN)
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test2)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = min_max_sc(tt_notarg_attn[i]*np.not_equal(sent_test2[0][i], 0))#*(sent_test2[2][i]+sent_test2[3][i]))
    tt2 = min_max_sc(tt_wttarg_attn[i]*np.not_equal(sent_test[0][i], 0))#*(sent_test[2][i]+sent_test[3][i]))
    
    axes[i][0].bar(x=tokenizer.convert_ids_to_tokens(sent_test2[0][i]), height=tt1, width=0.2)
    axes[i][1].bar(x=tokenizer.convert_ids_to_tokens(sent_test[0][i]), height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)
#     axes[i][0].set_xticks(range(sent_test2[0][i]+1))
#     axes[i][1].set_xticks(range(sent_test[0][i]+1))
#     axes[i][0].set_xticklabels(sent_test2[1][i])
#     axes[i][1].set_xticklabels(sent_test[1][i])
```

```python

```

<!-- #region {"colab_type": "text", "id": "mWvVHmCSzShW"} -->
## antonym
- w.o. target word: more emphasize on yummy? (sentences are short...)
<!-- #endregion -->

```python
tt_sent = sentences_high_ant
tt_test  = convert_text_to_examples(tt_sent[0], tt_sent[1], np.zeros(len(tt_sent[0])))
sent_test  = convert_examples_to_features(tokenizer, tt_test, True, MAX_SEQ_LEN)
sent_test2 = convert_examples_to_features(tokenizer, tt_test, False, MAX_SEQ_LEN)
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test2)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = min_max_sc(tt_notarg_attn[i]*np.not_equal(sent_test2[0][i], 0))#*(sent_test2[2][i]+sent_test2[3][i]))
    tt2 = min_max_sc(tt_wttarg_attn[i]*np.not_equal(sent_test[0][i], 0))#*(sent_test[2][i]+sent_test[3][i]))
    
    axes[i][0].bar(x=tokenizer.convert_ids_to_tokens(sent_test2[0][i]), height=tt1, width=0.2)
    axes[i][1].bar(x=tokenizer.convert_ids_to_tokens(sent_test[0][i]), height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)
    
```

<!-- #region {"colab_type": "text", "id": "pfnzJsSVzbyP"} -->
## causal
- w.o. target word: more emphasize on lying, which is the cause
<!-- #endregion -->

```python
tt_sent = sentences_high_cau
tt_test  = convert_text_to_examples(tt_sent[0], tt_sent[1], np.zeros(len(tt_sent[0])))
sent_test  = convert_examples_to_features(tokenizer, tt_test, True, MAX_SEQ_LEN)
sent_test2 = convert_examples_to_features(tokenizer, tt_test, False, MAX_SEQ_LEN)
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test2)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = min_max_sc(tt_notarg_attn[i]*np.not_equal(sent_test2[0][i], 0))#*(sent_test2[2][i]+sent_test2[3][i]))
    tt2 = min_max_sc(tt_wttarg_attn[i]*np.not_equal(sent_test[0][i], 0))#*(sent_test[2][i]+sent_test[3][i]))
    
    axes[i][0].bar(x=tokenizer.convert_ids_to_tokens(sent_test2[0][i]), height=tt1, width=0.2)
    axes[i][1].bar(x=tokenizer.convert_ids_to_tokens(sent_test[0][i]), height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)    
```

```python colab={} colab_type="code" id="BuzZxn09zght"

```

<!-- #region {"colab_type": "text", "id": "0oi5OecG6tH3"} -->
## homonym
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} colab_type="code" executionInfo={"elapsed": 29988, "status": "ok", "timestamp": 1569439385379, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="Y-X1vP276s0y" outputId="e779d05b-e2c5-4860-a419-dac0d8352ae2"
df_ex_homopoly = pd.read_csv("./dataset/polynym_homonym.csv")
df_ex_homopoly
```

```python colab={"base_uri": "https://localhost:8080/", "height": 595} colab_type="code" executionInfo={"elapsed": 29981, "status": "ok", "timestamp": 1569439385380, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="s8lZFJn69trs" outputId="9de45f8a-3364-4937-833b-cbc3a37fbadc"
sentences_hp = proc_sentences(df_ex_homopoly, "sentence", "targ")
sentences_hp2 = proc_sentences(df_ex_homopoly, "sentence", None)
sentences_hp2
```

```python colab={} colab_type="code" id="wtE2eSxT9trv"
sentences_homo = [xx[df_ex_homopoly["type"]=="homonymy"] for xx in sentences_hp]
sentences_poly = [xx[df_ex_homopoly["type"]=="polysemy"] for xx in sentences_hp]
sentences_homo2 = [xx[df_ex_homopoly["type"]=="homonymy"] for xx in sentences_hp2]
sentences_poly2 = [xx[df_ex_homopoly["type"]=="polysemy"] for xx in sentences_hp2]
```

```python colab={} colab_type="code" id="wFmRkvNp9egL"

```

```python
tt_sent = sentences_homo
tt_test  = convert_text_to_examples(tt_sent[0], tt_sent[1], np.zeros(len(tt_sent[0])))
sent_test  = convert_examples_to_features(tokenizer, tt_test, True, MAX_SEQ_LEN)
sent_test2 = convert_examples_to_features(tokenizer, tt_test, False, MAX_SEQ_LEN)
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test2)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = min_max_sc(tt_notarg_attn[i]*np.not_equal(sent_test2[0][i], 0))#*(sent_test2[2][i]+sent_test2[3][i]))
    tt2 = min_max_sc(tt_wttarg_attn[i]*np.not_equal(sent_test[0][i], 0))#*(sent_test[2][i]+sent_test[3][i]))
    
    axes[i][0].bar(x=tokenizer.convert_ids_to_tokens(sent_test2[0][i]), height=tt1, width=0.2)
    axes[i][1].bar(x=tokenizer.convert_ids_to_tokens(sent_test[0][i]), height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)
    
```

```python colab={} colab_type="code" id="j6Itr67w-CLB"

```

<!-- #region {"colab_type": "text", "id": "t7Vw08C-Hpee"} -->
## polysenym
<!-- #endregion -->

```python
tt_sent = sentences_poly
tt_test  = convert_text_to_examples(tt_sent[0], tt_sent[1], np.zeros(len(tt_sent[0])))
sent_test  = convert_examples_to_features(tokenizer, tt_test, True, MAX_SEQ_LEN)
sent_test2 = convert_examples_to_features(tokenizer, tt_test, False, MAX_SEQ_LEN)
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test2)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = min_max_sc(tt_notarg_attn[i]*np.not_equal(sent_test2[0][i], 0))#*(sent_test2[2][i]+sent_test2[3][i]))
    tt2 = min_max_sc(tt_wttarg_attn[i]*np.not_equal(sent_test[0][i], 0))#*(sent_test[2][i]+sent_test[3][i]))
    
    axes[i][0].bar(x=tokenizer.convert_ids_to_tokens(sent_test2[0][i]), height=tt1, width=0.2)
    axes[i][1].bar(x=tokenizer.convert_ids_to_tokens(sent_test[0][i]), height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)
    
```

```python

```
