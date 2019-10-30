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
from models.build_models import *
from models.train_models import *
from utils.utils import *
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
# sentences_wttarg = proc_sentences(df_cloze, 'sentence', 'syn1')
# sentences_notarg = proc_sentences(df_cloze, 'sentence', None)
sentences = proc_sentences_dscovar(df_cloze, 'sentence', 'targ', 'bert')
```

```python
sentences
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
tt = convert_examples_to_features(tokenizer, train_examples[:1], False, MAX_SEQ_LEN)
tt[:4]
```

<!-- #region {"colab_type": "text", "id": "l4rgsjFnxP3U", "toc-hr-collapsed": false} -->
# BERT + Attention model
<!-- #endregion -->

## separate targ-cntx layers

```python
K.clear_session()
sess = tf.Session()

model = build_model_bert(MAX_SEQ_LEN, finetune_emb=True, attention_layer=True, sep_cntx_targ=True)
initialize_vars(sess)

model.summary()
```

```python
plot_model(model)
```

## Single targ-cntx layers

```python
K.clear_session()
sess = tf.Session()

model = build_model_bert(MAX_SEQ_LEN, finetune_emb=False, attention_layer=True, sep_cntx_targ=False)
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

## /w attention (finetune BERT)

```python
NUM_ITER = [2,3,5]
LEARNING_RATE = [1e-3, 1e-4, 5e-5, 3e-5, 1e-5]
BATCH_SIZE = [16, 32]
```

### 1emb 

```python
# _l_rate = 5e-5
# _num_iter = 3
# _batch_size = 16

for _l_rate in LEARNING_RATE:
    for _batch_size in BATCH_SIZE:
        for _num_iter in NUM_ITER:
            gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
            train_bertmod_cv(X, y, False,
                             gkf_split, True, True, False, 
                             "./model_weights/finetune/bert/1emb/model_bert_notarg_wtattn_"+y_type+"_cvTwrd"+"_i"+str(_num_iter)+"_b"+str(_batch_size)+"_lr"+str(_l_rate),
                             "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_"+y_type+"_cvTwrd"+"_i"+str(_num_iter)+"_b"+str(_batch_size)+"_lr"+str(_l_rate),
                             MAX_SEQ_LEN, _l_rate, _num_iter, _batch_size)
```

### 2emb 

```python
NUM_ITER = [2,3,5]
LEARNING_RATE = [5e-5, 3e-5, 1e-5]
BATCH_SIZE = [16] #32: OOM error
```

```python
for _l_rate in LEARNING_RATE:
    for _batch_size in BATCH_SIZE:
        for _num_iter in NUM_ITER:
            gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
            train_bertmod_cv(X, y, False,
                             gkf_split, True, True, True, 
                             "./model_weights/finetune/bert/2emb/model_bert_notarg_wtattn_"+y_type+"_cvTwrd"+"_i"+str(_num_iter)+"_b"+str(_batch_size)+"_lr"+str(_l_rate),
                             "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_"+y_type+"_cvTwrd"+"_i"+str(_num_iter)+"_b"+str(_batch_size)+"_lr"+str(_l_rate),
                             MAX_SEQ_LEN, _l_rate, _num_iter, _batch_size)
```

```python

```

# Classification performance 

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
fig, axes = plt.subplots(ncols=6, figsize=(42, 6))
tt_col = sns.color_palette("colorblind", 6)

# iter: 5; batch: 16
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i5_b16_lr0.001*",  "bert_notarg_wtattn_bws_1emb_lr1e-03", 0.50, "high", fig, axes[0], tt_col[0], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i5_b16_lr0.0001*", "bert_notarg_wtattn_bws_1emb_lr1e-04", 0.50, "high", fig, axes[0], tt_col[1], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i5_b16_lr5e-05*",  "bert_notarg_wtattn_bws_1emb_lr5e-05", 0.50, "high", fig, axes[0], tt_col[2], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i5_b16_lr3e-05*",  "bert_notarg_wtattn_bws_1emb_lr3e-05", 0.50, "high", fig, axes[0], tt_col[3], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i5_b16_lr1e-05*",  "bert_notarg_wtattn_bws_1emb_lr1e-05", 0.50, "high", fig, axes[0], tt_col[4], '-')

# iter: 5; batch: 16
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr0.001*",  "bert_notarg_wtattn_bws_2emb_lr1e-03", 0.50, "high", fig, axes[1], tt_col[0], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr0.0001*", "bert_notarg_wtattn_bws_2emb_lr1e-04", 0.50, "high", fig, axes[1], tt_col[1], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr5e-05*",  "bert_notarg_wtattn_bws_2emb_lr5e-05", 0.50, "high", fig, axes[1], tt_col[2], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr3e-05*",  "bert_notarg_wtattn_bws_2emb_lr3e-05", 0.50, "high", fig, axes[1], tt_col[3], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr1e-05*",  "bert_notarg_wtattn_bws_2emb_lr1e-05", 0.50, "high", fig, axes[1], tt_col[4], '-')

# iter: 3: batch: 16
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr0.001*",  "bert_notarg_wtattn_bws_1emb_lr1e-03", 0.50, "high", fig, axes[2], tt_col[0], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr0.0001*", "bert_notarg_wtattn_bws_1emb_lr1e-04", 0.50, "high", fig, axes[2], tt_col[1], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr5e-05*",  "bert_notarg_wtattn_bws_1emb_lr5e-05", 0.50, "high", fig, axes[2], tt_col[2], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr3e-05*",  "bert_notarg_wtattn_bws_1emb_lr3e-05", 0.50, "high", fig, axes[2], tt_col[3], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr1e-05*",  "bert_notarg_wtattn_bws_1emb_lr1e-05", 0.50, "high", fig, axes[2], tt_col[4], '-')

# iter: 3; batch: 16
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr0.001*",  "bert_notarg_wtattn_bws_2emb_lr1e-03", 0.50, "high", fig, axes[3], tt_col[0], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr0.0001*", "bert_notarg_wtattn_bws_2emb_lr1e-04", 0.50, "high", fig, axes[3], tt_col[1], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr5e-05*",  "bert_notarg_wtattn_bws_2emb_lr5e-05", 0.50, "high", fig, axes[3], tt_col[2], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr3e-05*",  "bert_notarg_wtattn_bws_2emb_lr3e-05", 0.50, "high", fig, axes[3], tt_col[3], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i3_b16_lr1e-05*",  "bert_notarg_wtattn_bws_2emb_lr1e-05", 0.50, "high", fig, axes[3], tt_col[4], '-')

# iter: 2: batch: 16
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr0.001*",  "bert_notarg_wtattn_bws_1emb_lr1e-03", 0.50, "high", fig, axes[4], tt_col[0], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr0.0001*", "bert_notarg_wtattn_bws_1emb_lr1e-04", 0.50, "high", fig, axes[4], tt_col[1], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr5e-05*",  "bert_notarg_wtattn_bws_1emb_lr5e-05", 0.50, "high", fig, axes[4], tt_col[2], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr3e-05*",  "bert_notarg_wtattn_bws_1emb_lr3e-05", 0.50, "high", fig, axes[4], tt_col[3], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/1emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr1e-05*",  "bert_notarg_wtattn_bws_1emb_lr1e-05", 0.50, "high", fig, axes[4], tt_col[4], '-')

# iter: 2; batch: 16
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr0.001*",  "bert_notarg_wtattn_bws_2emb_lr1e-03", 0.50, "high", fig, axes[5], tt_col[0], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr0.0001*", "bert_notarg_wtattn_bws_2emb_lr1e-04", 0.50, "high", fig, axes[5], tt_col[1], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr5e-05*",  "bert_notarg_wtattn_bws_2emb_lr5e-05", 0.50, "high", fig, axes[5], tt_col[2], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr3e-05*",  "bert_notarg_wtattn_bws_2emb_lr3e-05", 0.50, "high", fig, axes[5], tt_col[3], '-')
roc_cv_plot(resp_bws_cvTwrd, "./model_predict/finetune/bert/2emb/preds_bert_notarg_wtattn_bws_cvTwrd_i2_b16_lr1e-05*",  "bert_notarg_wtattn_bws_2emb_lr1e-05", 0.50, "high", fig, axes[5], tt_col[4], '-')

axes[0].set_title("iter: 5; batch: 16")
axes[1].set_title("iter: 5; batch: 16")
axes[2].set_title("iter: 3; batch: 16")
axes[3].set_title("iter: 3; batch: 16")
axes[4].set_title("iter: 2; batch: 16")
axes[5].set_title("iter: 2; batch: 16")
```

```python

```
