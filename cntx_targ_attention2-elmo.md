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


from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
from layers.embeddings import ElmoLayer
from layers.attention import AttentionLayer
from models.build_models import build_model_elmo, initialize_vars
from models.train_models import train_elmomod_cv
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
MAX_SEQ_LEN = 20
FILTERS = '"#$%&()*+/:;=@[\\]^_`{|}~\t\n'
```

```python colab={} colab_type="code" id="Ac6_DVsSNPo6"
def proc_sentences(df, col_sentence, col_targ, include_target=True):
    sentences = []
    li_mask_LH = []
    li_mask_RH = []
    li_mask_cntx = []
#     li_targ = []
    li_targ_idx = []
    li_sent_len = []
    li_sent_pad = []
    for i in range(df.shape[0]):
        sent = df[col_sentence][i]
        targ = df[col_targ][i]
        if(include_target):
            targ = df[col_targ][i]
        else:
            targ = "<UNK>"
        sent = sent.replace("______", targ)
        sent = sent.replace("<BOS>", "").replace(".", " .").replace(",", " ,").replace("!", " !").replace(",?", " ?").replace("'s", " 's")
        sent_tok = text_to_word_sequence(sent, lower=False, filters=FILTERS)
        sent_pad = pad_sequences([sent_tok], maxlen=MAX_SEQ_LEN, dtype='object', padding='post', value=[""])       
        targ_idx = np.where(targ==sent_pad[0])[0][0]
        
        mask_LH = [0]*(MAX_SEQ_LEN)
        mask_RH = [0]*(MAX_SEQ_LEN)
        for i in range(targ_idx):
            mask_LH[i] = 1
        for i in range(targ_idx+1, len(sent_tok)):
            mask_RH[i] = 1
        
        sent_len = len(sent_tok)
        
#         li_targ.append(targ)
        li_targ_idx.append(targ_idx)
        li_sent_len.append(sent_len)
        li_sent_pad.append(list(sent_pad)[0])
        li_mask_LH.append(mask_LH)
        li_mask_RH.append(mask_RH)
#     sentences = [np.array(li_targ_idx), np.array(li_sent_len), np.array(li_sent_pad), np.array(li_targ)]
#     sentences = [np.array(li_targ_idx), np.array(li_sent_len), np.array(li_sent_pad)]
    sentences = [np.array(li_sent_len), np.array(li_sent_pad), # np.array(li_targ_idx),
                 np.array(li_mask_LH), np.array(li_mask_RH)] #, np.sum((li_mask_LH, li_mask_RH), axis=0)]
    return(sentences)
```

```python colab={} colab_type="code" id="Ac6_DVsSNPo6"
sentences = proc_sentences(df_cloze, 'sentence', 'targ', True)
sentences_notarg = proc_sentences(df_cloze, 'sentence', 'targ', False)
```

```python
sentences
```

```python
sentences_notarg
```

```python colab={} colab_type="code" id="IZUg96JlhwqB"
mm_scaler = MinMaxScaler()

resp_scores = mm_scaler.fit_transform(df_cloze[['ent_cloze', 'elmo_score', 'scores_sum']])
resp_lex = resp_scores[:, 0]
resp_lex = 1-resp_lex # reversing the direction: high score for high informative sentences

resp_sem = resp_scores[:, 1]
resp_bws = resp_scores[:, 2]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 566} colab_type="code" executionInfo={"elapsed": 14111, "status": "ok", "timestamp": 1569434167349, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="dMBnF8u4RUm9" outputId="8ba84efe-5ea6-4f04-829f-797a89d9c9e7"
sns.pairplot(pd.DataFrame({"resp_lex":resp_lex, "resp_sem":resp_sem, "resp_bws":resp_bws}))
```

<!-- #region {"colab_type": "text", "id": "l4rgsjFnxP3U", "toc-hr-collapsed": false} -->
# ELMo + Attention model
<!-- #endregion -->

```python
K.clear_session()
sess = tf.Session()

model = build_model_elmo(MAX_SEQ_LEN, attention_layer=True)
initialize_vars(sess)

model.summary()
```

```python
plot_model(model)
```

# K-fold training and predictions 

```python
_num_iter = 20
_batch_size = 64
```

## fold: target words

```python
gkf1 = GroupKFold(n_splits=K_FOLDS)
```

```python
X = sentences
X_notarg = sentences_notarg
y = resp_bws
y_type = 'bws'
```

### including target word

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
```

```python
train_elmomod_cv(X, y, 
                 gkf_split, False,
                 "./model_weights/model_elmo_wttarg_noattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_elmo_wttarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_elmomod_cv(X, y, 
                 gkf_split, True,
                 "./model_weights/model_elmo_wttarg_wtattn_"+y_type+"_cvTwrd", 
                 "./model_predict/preds_elmo_wttarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_elmomod_cv(X_notarg, y,
                 gkf_split, False,
                 "./model_weights/model_elmo_notarg_noattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_elmo_notarg_noattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf1.split(df_cloze['sentence'], groups=df_cloze['targ'])
train_elmomod_cv(X_notarg, y,
                 gkf_split, True,
                 "./model_weights/model_elmo_notarg_wtattn_"+y_type+"_cvTwrd",  
                 "./model_predict/preds_elmo_notarg_wtattn_"+y_type+"_cvTwrd",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

## fold: target word locations

```python
targ_loc_cat = []
for i in range(df_cloze.shape[0]):
    if(df_cloze['targ_loc_end'][i]): 
        targ_loc_cat.append('last')
    else:
        if(df_cloze['targ_loc_rel'][i] <= 0.5):
            targ_loc_cat.append('less_50')
        elif((df_cloze['targ_loc_rel'][i] > 0.5)&(df_cloze['targ_loc_rel'][i] <= 0.65)):
            targ_loc_cat.append("(50_65]")
        elif((df_cloze['targ_loc_rel'][i] > 0.65)):
            targ_loc_cat.append("more_65")            
```

```python
Counter(targ_loc_cat)
```

```python
gkf2 = GroupKFold(n_splits=len(Counter(targ_loc_cat)))
```

### including target word

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_elmomod_cv(X, y,
                 gkf_split, False,
                 "./model_weights/model_elmo_wttarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_elmo_wttarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_elmomod_cv(X, y,
                 gkf_split, True,
                 "./model_weights/model_elmo_wttarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_elmo_wttarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

### target word as oov token

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_elmomod_cv(X_notarg, y, 
                 gkf_split, False,
                 "./model_weights/model_elmo_notarg_noattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_elmo_notarg_noattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python
gkf_split = gkf2.split(df_cloze['sentence'], groups=targ_loc_cat)
train_elmomod_cv(X_notarg, y,
                 gkf_split, True,
                 "./model_weights/model_elmo_notarg_wtattn_"+y_type+"_cvTloc", 
                 "./model_predict/preds_elmo_notarg_wtattn_"+y_type+"_cvTloc",
                 MAX_SEQ_LEN, _num_iter, _batch_size)
```

```python

```

# Classification performance 

```python
def roc_cv(cv_true_scores, pred_score_file_loc, score_type, cut, direction, fig, ax, col):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    pred_score_files = sorted(glob.glob(pred_score_file_loc))

    for i, cv_true_score in enumerate(cv_true_scores):
        if(direction=="high"):
            fpr, tpr, _ = roc_curve(cv_true_score > np.quantile(cv_true_score, q=[cut]), np.load(pred_score_files[i]))
        if(direction=="low"):
            fpr, tpr, _ = roc_curve(cv_true_score < np.quantile(cv_true_score, q=[cut]), 1-np.load(pred_score_files[i]))
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
             color=col, alpha=1,
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
sent_test_cvTwrd = []
sent_test_cvTloc = []
resp_bws_cvTwrd = []
resp_bws_cvTloc = []
resp_lex_cvTwrd = []
resp_lex_cvTloc = []
resp_sem_cvTwrd = []
resp_sem_cvTloc = []


for train_idx, test_idx in gkf1.split(df_cloze['sentence'], groups=df_cloze['targ']):
    sent_test_cvTwrd.append([sent[test_idx] for sent in sentences])
    resp_bws_cvTwrd.append([resp_bws[i] for i in test_idx])
    resp_sem_cvTwrd.append([resp_sem[i] for i in test_idx])
    resp_lex_cvTwrd.append([resp_lex[i] for i in test_idx])
    
for train_idx, test_idx in gkf2.split(df_cloze['sentence'], groups=targ_loc_cat):
    sent_test_cvTloc.append([sent[test_idx] for sent in sentences])
    resp_bws_cvTloc.append([resp_bws[i] for i in test_idx])
    resp_sem_cvTloc.append([resp_sem[i] for i in test_idx])
    resp_lex_cvTloc.append([resp_lex[i] for i in test_idx])
```

## CV over target words vs. target word location

```python
Counter(targ_loc_cat)
```

### fitted to: BWS score
- target location may bias the prediction results in some cases, but not by much
- attention layer does not improve the performance
- including a target word does not help 

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTwrd*", "elmo_wttarg_noattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[0])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTwrd*", "elmo_wttarg_wtattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[1])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_bws_cvTwrd*", "elmo_notarg_noattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[2])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTwrd*", "elmo_notarg_wtattn_bws_cvTwrd", 0.50, "high", fig, axes[0], tt_col[3])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTloc*", "elmo_wttarg_noattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[0])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTloc*", "elmo_wttarg_wtattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[1])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_noattn_bws_cvTloc*", "elmo_notarg_noattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[2])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTloc*", "elmo_notarg_wtattn_bws_cvTloc", 0.50, "high", fig, axes[0], tt_col[3])

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTwrd*", "elmo_wttarg_noattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[0])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTwrd*", "elmo_wttarg_wtattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[1])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_bws_cvTwrd*", "elmo_notarg_noattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[2])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTwrd*", "elmo_notarg_wtattn_bws_cvTwrd", 0.25, "high", fig, axes[1], tt_col[3])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTloc*", "elmo_wttarg_noattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[0])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTloc*", "elmo_wttarg_wtattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[1])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_noattn_bws_cvTloc*", "elmo_notarg_noattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[2])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTloc*", "elmo_notarg_wtattn_bws_cvTloc", 0.25, "high", fig, axes[1], tt_col[3])

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTwrd*", "elmo_wttarg_noattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[0])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTwrd*", "elmo_wttarg_wtattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[1])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_bws_cvTwrd*", "elmo_notarg_noattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[2])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTwrd*", "elmo_notarg_wtattn_bws_cvTwrd", 0.10, "high", fig, axes[2], tt_col[3])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTloc*", "elmo_wttarg_noattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[0])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTloc*", "elmo_wttarg_wtattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[1])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_noattn_bws_cvTloc*", "elmo_notarg_noattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[2])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTloc*", "elmo_notarg_wtattn_bws_cvTloc", 0.10, "high", fig, axes[2], tt_col[3])

```

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTwrd*", "elmo_wttarg_noattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[0])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTwrd*", "elmo_wttarg_wtattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[1])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_bws_cvTwrd*", "elmo_notarg_noattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[2])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTwrd*", "elmo_notarg_wtattn_bws_cvTwrd", 0.50, "low", fig, axes[0], tt_col[3])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTloc*", "elmo_wttarg_noattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[0])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTloc*", "elmo_wttarg_wtattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[1])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_noattn_bws_cvTloc*", "elmo_notarg_noattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[2])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTloc*", "elmo_notarg_wtattn_bws_cvTloc", 0.50, "low", fig, axes[0], tt_col[3])

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTwrd*", "elmo_wttarg_noattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[0])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTwrd*", "elmo_wttarg_wtattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[1])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_bws_cvTwrd*", "elmo_notarg_noattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[2])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTwrd*", "elmo_notarg_wtattn_bws_cvTwrd", 0.25, "low", fig, axes[1], tt_col[3])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTloc*", "elmo_wttarg_noattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[0])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTloc*", "elmo_wttarg_wtattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[1])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_noattn_bws_cvTloc*", "elmo_notarg_noattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[2])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTloc*", "elmo_notarg_wtattn_bws_cvTloc", 0.25, "low", fig, axes[1], tt_col[3])

roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTwrd*", "elmo_wttarg_noattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[0])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTwrd*", "elmo_wttarg_wtattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[1])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_bws_cvTwrd*", "elmo_notarg_noattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[2])
roc_cv(resp_bws_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTwrd*", "elmo_notarg_wtattn_bws_cvTwrd", 0.10, "low", fig, axes[2], tt_col[3])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_bws_cvTloc*", "elmo_wttarg_noattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[0])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_bws_cvTloc*", "elmo_wttarg_wtattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[1])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_noattn_bws_cvTloc*", "elmo_notarg_noattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[2])
roc_cv(resp_bws_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_bws_cvTloc*", "elmo_notarg_wtattn_bws_cvTloc", 0.10, "low", fig, axes[2], tt_col[3])

```

### fitted to: Semantic distance
- target location may bias the prediction results in some cases, but not by much
- attention layer maybe helpful in some cases
- including a target word does not help 

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTwrd*", "elmo_wttarg_noattn_sem_cvTwrd", 0.50, "high", fig, axes[0], tt_col[0])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTwrd*", "elmo_wttarg_wtattn_sem_cvTwrd", 0.50, "high", fig, axes[0], tt_col[1])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_sem_cvTwrd*", "elmo_notarg_noattn_sem_cvTwrd", 0.50, "high", fig, axes[0], tt_col[2])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTwrd*", "elmo_notarg_wtattn_sem_cvTwrd", 0.50, "high", fig, axes[0], tt_col[3])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTloc*", "elmo_wttarg_noattn_sem_cvTloc", 0.50, "high", fig, axes[0], tt_col[0])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTloc*", "elmo_wttarg_wtattn_sem_cvTloc", 0.50, "high", fig, axes[0], tt_col[1])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_noattn_sem_cvTloc*", "elmo_notarg_noattn_sem_cvTloc", 0.50, "high", fig, axes[0], tt_col[2])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTloc*", "elmo_notarg_wtattn_sem_cvTloc", 0.50, "high", fig, axes[0], tt_col[3])

roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTwrd*", "elmo_wttarg_noattn_sem_cvTwrd", 0.25, "high", fig, axes[1], tt_col[0])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTwrd*", "elmo_wttarg_wtattn_sem_cvTwrd", 0.25, "high", fig, axes[1], tt_col[1])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_sem_cvTwrd*", "elmo_notarg_noattn_sem_cvTwrd", 0.25, "high", fig, axes[1], tt_col[2])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTwrd*", "elmo_notarg_wtattn_sem_cvTwrd", 0.25, "high", fig, axes[1], tt_col[3])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTloc*", "elmo_wttarg_noattn_sem_cvTloc", 0.25, "high", fig, axes[1], tt_col[0])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTloc*", "elmo_wttarg_wtattn_sem_cvTloc", 0.25, "high", fig, axes[1], tt_col[1])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_noattn_sem_cvTloc*", "elmo_notarg_noattn_sem_cvTloc", 0.25, "high", fig, axes[1], tt_col[2])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTloc*", "elmo_notarg_wtattn_sem_cvTloc", 0.25, "high", fig, axes[1], tt_col[3])

roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTwrd*", "elmo_wttarg_noattn_sem_cvTwrd", 0.10, "high", fig, axes[2], tt_col[0])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTwrd*", "elmo_wttarg_wtattn_sem_cvTwrd", 0.10, "high", fig, axes[2], tt_col[1])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_sem_cvTwrd*", "elmo_notarg_noattn_sem_cvTwrd", 0.10, "high", fig, axes[2], tt_col[2])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTwrd*", "elmo_notarg_wtattn_sem_cvTwrd", 0.10, "high", fig, axes[2], tt_col[3])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTloc*", "elmo_wttarg_noattn_sem_cvTloc", 0.10, "high", fig, axes[2], tt_col[0])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTloc*", "elmo_wttarg_wtattn_sem_cvTloc", 0.10, "high", fig, axes[2], tt_col[1])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_noattn_sem_cvTloc*", "elmo_notarg_noattn_sem_cvTloc", 0.10, "high", fig, axes[2], tt_col[2])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTloc*", "elmo_notarg_wtattn_sem_cvTloc", 0.10, "high", fig, axes[2], tt_col[3])

```

```python
fig, axes = plt.subplots(ncols=3, figsize=(24, 6))
tt_col = sns.color_palette("colorblind", 6)

roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTwrd*", "elmo_wttarg_noattn_sem_cvTwrd", 0.50, "low", fig, axes[0], tt_col[0])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTwrd*", "elmo_wttarg_wtattn_sem_cvTwrd", 0.50, "low", fig, axes[0], tt_col[1])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_sem_cvTwrd*", "elmo_notarg_noattn_sem_cvTwrd", 0.50, "low", fig, axes[0], tt_col[2])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTwrd*", "elmo_notarg_wtattn_sem_cvTwrd", 0.50, "low", fig, axes[0], tt_col[3])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTloc*", "elmo_wttarg_noattn_sem_cvTloc", 0.50, "low", fig, axes[0], tt_col[0])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTloc*", "elmo_wttarg_wtattn_sem_cvTloc", 0.50, "low", fig, axes[0], tt_col[1])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_noattn_sem_cvTloc*", "elmo_notarg_noattn_sem_cvTloc", 0.50, "low", fig, axes[0], tt_col[2])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTloc*", "elmo_notarg_wtattn_sem_cvTloc", 0.50, "low", fig, axes[0], tt_col[3])

roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTwrd*", "elmo_wttarg_noattn_sem_cvTwrd", 0.25, "low", fig, axes[1], tt_col[0])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTwrd*", "elmo_wttarg_wtattn_sem_cvTwrd", 0.25, "low", fig, axes[1], tt_col[1])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_sem_cvTwrd*", "elmo_notarg_noattn_sem_cvTwrd", 0.25, "low", fig, axes[1], tt_col[2])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTwrd*", "elmo_notarg_wtattn_sem_cvTwrd", 0.25, "low", fig, axes[1], tt_col[3])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTloc*", "elmo_wttarg_noattn_sem_cvTloc", 0.25, "low", fig, axes[1], tt_col[0])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTloc*", "elmo_wttarg_wtattn_sem_cvTloc", 0.25, "low", fig, axes[1], tt_col[1])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_noattn_sem_cvTloc*", "elmo_notarg_noattn_sem_cvTloc", 0.25, "low", fig, axes[1], tt_col[2])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTloc*", "elmo_notarg_wtattn_sem_cvTloc", 0.25, "low", fig, axes[1], tt_col[3])

roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTwrd*", "elmo_wttarg_noattn_sem_cvTwrd", 0.10, "low", fig, axes[2], tt_col[0])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTwrd*", "elmo_wttarg_wtattn_sem_cvTwrd", 0.10, "low", fig, axes[2], tt_col[1])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_noattn_sem_cvTwrd*", "elmo_notarg_noattn_sem_cvTwrd", 0.10, "low", fig, axes[2], tt_col[2])
roc_cv(resp_sem_cvTwrd, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTwrd*", "elmo_notarg_wtattn_sem_cvTwrd", 0.10, "low", fig, axes[2], tt_col[3])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_noattn_sem_cvTloc*", "elmo_wttarg_noattn_sem_cvTloc", 0.10, "low", fig, axes[2], tt_col[0])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_wttarg_wtattn_sem_cvTloc*", "elmo_wttarg_wtattn_sem_cvTloc", 0.10, "low", fig, axes[2], tt_col[1])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_noattn_sem_cvTloc*", "elmo_notarg_noattn_sem_cvTloc", 0.10, "low", fig, axes[2], tt_col[2])
roc_cv(resp_sem_cvTloc, "./model_predict/preds_elmo_notarg_wtattn_sem_cvTloc*", "elmo_notarg_wtattn_sem_cvTloc", 0.10, "low", fig, axes[2], tt_col[3])

```

# Prediction results 

```python
K.clear_session()
sess = tf.Session()

model_notarg = build_model_elmo(MAX_SEQ_LEN, True)
model_wttarg = build_model_elmo(MAX_SEQ_LEN, True)
initialize_vars(sess)

model_notarg.load_weights("model_weights/model_elmo_notarg_wtattn_sem_cvTwrd0.h5")
model_wttarg.load_weights("model_weights/model_elmo_wttarg_wtattn_sem_cvTwrd0.h5")
```

## score predictions 

```python
# attn_bws_pred_test = np.reshape(model.predict(sent_test_cvTwrd[0], batch_size=128), -1)
notarg_wtattn_pred_test = np.reshape(model_notarg.predict(sent_test_cvTwrd[0], batch_size=128), -1)
wttarg_wtattn_pred_test = np.reshape(model_wttarg.predict(sent_test_cvTwrd[0], batch_size=128), -1)
```

### for fold 0

```python
plt.figure(figsize=(6,6))
sns.scatterplot(resp_sem_cvTwrd[0], notarg_wtattn_pred_test, alpha=0.3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("SEM: Observed")
plt.ylabel("SEM: notarg_Predicted")
print(spearmanr(resp_sem_cvTwrd[0], notarg_wtattn_pred_test))
print(mean_absolute_error(resp_sem_cvTwrd[0], notarg_wtattn_pred_test))
```

```python
plt.figure(figsize=(6,6))
sns.scatterplot(resp_sem_cvTwrd[0], wttarg_wtattn_pred_test, alpha=0.3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("SEM: Observed")
plt.ylabel("SEM: notarg_Predicted")
print(spearmanr(resp_sem_cvTwrd[0], wttarg_wtattn_pred_test))
print(mean_absolute_error(resp_sem_cvTwrd[0], wttarg_wtattn_pred_test))
```

```python
sns.scatterplot(notarg_wtattn_pred_test, wttarg_wtattn_pred_test, alpha=0.3)
```

### consolidating predictions from all cv folds

```python
tt_files = sorted(glob.glob("./model_predict/preds_elmo_notarg_wtattn_sem_cvTwrd*"))
tt_preds = [np.load(f) for f in tt_files]
tt_preds = [x for xx in tt_preds for x in xx]
tt_obs = sum(resp_sem_cvTwrd, [])

plt.figure(figsize=(6,6))
sns.scatterplot(tt_obs, tt_preds, alpha=0.3)
print(spearmanr(tt_obs, tt_preds))
print(mean_absolute_error(tt_obs, tt_preds))
```

```python
tt_files = sorted(glob.glob("./model_predict/preds_elmo_wttarg_wtattn_sem_cvTwrd*"))
tt_preds = [np.load(f) for f in tt_files]
tt_preds = [x for xx in tt_preds for x in xx]
tt_obs = sum(resp_sem_cvTwrd, [])

plt.figure(figsize=(6,6))
sns.scatterplot(tt_obs, tt_preds, alpha=0.3)
print(spearmanr(tt_obs, tt_preds))
print(mean_absolute_error(tt_obs, tt_preds))
```

```python

```

## attention interpretations

```python colab={} colab_type="code" id="1zpBhTwiyaT_"
model_notarg_sfmxres = Model(model_notarg.inputs, model_notarg.get_layer('attention_layer').output[1])
model_wttarg_sfmxres = Model(model_wttarg.inputs, model_wttarg.get_layer('attention_layer').output[1])
# initialize_vars(sess)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 2458085, "status": "ok", "timestamp": 1569436612264, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="r_nzXPiQbu61" outputId="c2c93a03-1d93-4d5b-ee63-64b9003a4b5a"
hi_test_idx15 = np.prod([(resp_bws_cvTwrd[0] > np.quantile(resp_bws_cvTwrd[0], q=[0.90])),
                         (resp_sem_cvTwrd[0] > np.quantile(resp_sem_cvTwrd[0], q=[0.90]))], axis=0)
lo_test_idx15 = np.prod([(resp_bws_cvTwrd[0] < np.quantile(resp_bws_cvTwrd[0], q=[0.13])),
                         (resp_sem_cvTwrd[0] < np.quantile(resp_sem_cvTwrd[0], q=[0.13]))], axis=0)
hi_test_idx15 = np.where(hi_test_idx15)[0]
lo_test_idx15 = np.where(lo_test_idx15)[0]
(hi_test_idx15, lo_test_idx15)
```

```python colab={} colab_type="code" id="XllPYkfLyfAZ"
tt_notarg_attn = model_notarg_sfmxres.predict([sent for sent in sent_test_cvTwrd[0]])[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict([sent for sent in sent_test_cvTwrd[0]])[:,0,:]
```

#### high informative sentences

```python colab={"base_uri": "https://localhost:8080/", "height": 926} colab_type="code" executionInfo={"elapsed": 2437023, "status": "ok", "timestamp": 1569436591149, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="OWieih9zypmg" outputId="68027728-3b66-451c-f017-1b88ae6636a8"
fig, axes = plt.subplots(len(hi_test_idx15), 2, figsize=(16, len(hi_test_idx15)*3))
fig.subplots_adjust(hspace=2)

for i,j in enumerate(hi_test_idx15):
    tt1 = tt_notarg_attn[j]*(sent_test_cvTwrd[0][2][j]+sent_test_cvTwrd[0][3][j])
    tt1 = (tt1-tt1.min())/(tt1.max()-tt1.min())    
    tt2 = tt_wttarg_attn[j]*(sent_test_cvTwrd[0][2][j]+sent_test_cvTwrd[0][3][j])
    tt2 = (tt2-tt2.min())/(tt2.max()-tt2.min())    
    axes[i][0].bar(x=sent_test_cvTwrd[0][1][j], height=tt1, width=0.2)
    axes[i][0].set_title("notarg "+"SEM score: "+str(round(resp_sem_cvTwrd[0][j], 3))+
                      ", pred score: "+str(round(notarg_wtattn_pred_test[j], 3)))
    axes[i][1].bar(x=sent_test_cvTwrd[0][1][j], height=tt2, width=0.2)
    axes[i][1].set_title("wttarg "+"SEM score: "+str(round(resp_sem_cvTwrd[0][j], 3))+
                      ", pred score: "+str(round(wttarg_wtattn_pred_test[j], 3)))    
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)
```

#### low informative sentences

```python colab={"base_uri": "https://localhost:8080/", "height": 926} colab_type="code" executionInfo={"elapsed": 2437023, "status": "ok", "timestamp": 1569436591149, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="OWieih9zypmg" outputId="68027728-3b66-451c-f017-1b88ae6636a8"
fig, axes = plt.subplots(len(lo_test_idx15), 2, figsize=(16, len(lo_test_idx15)*3))
fig.subplots_adjust(hspace=2)

for i,j in enumerate(lo_test_idx15):
    tt1 = tt_notarg_attn[j]*(sent_test_cvTwrd[0][2][j]+sent_test_cvTwrd[0][3][j])
    tt1 = (tt1-tt1.min())/(tt1.max()-tt1.min())    
    tt2 = tt_wttarg_attn[j]*(sent_test_cvTwrd[0][2][j]+sent_test_cvTwrd[0][3][j])
    tt2 = (tt2-tt2.min())/(tt2.max()-tt2.min())    
    
    axes[i][0].bar(x=sent_test_cvTwrd[0][1][j], height=tt1, width=0.2)
    axes[i][0].set_title("notarg "+"SEM score: "+str(round(resp_sem_cvTwrd[0][j], 3))+
                      ", pred score: "+str(round(notarg_wtattn_pred_test[j], 3)))
    axes[i][1].bar(x=sent_test_cvTwrd[0][1][j], height=tt2, width=0.2)
    axes[i][1].set_title("wttarg "+"SEM score: "+str(round(resp_sem_cvTwrd[0][j], 3))+
                      ", pred score: "+str(round(wttarg_wtattn_pred_test[j], 3)))    
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)
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
sentences_high = proc_sentences(df_ex_hinfo, "Example context", "targ")
```

```python colab={} colab_type="code" id="pt_hk9VrxXjI"
sentences_high_syn = [xx[df_ex_hinfo["Cue "]=="Synonym"] for xx in sentences_high]
sentences_high_ant = [xx[df_ex_hinfo["Cue "]=="Antonym"] for xx in sentences_high]
sentences_high_cau = [xx[df_ex_hinfo["Cue "]=="Causal"]  for xx in sentences_high]
```

<!-- #region {"colab_type": "text", "id": "Ji05EQlln8zt"} -->
## synonym
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 902} colab_type="code" executionInfo={"elapsed": 11361, "status": "ok", "timestamp": 1569439342415, "user": {"displayName": "Sungjin Nam", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBm-uC84SkKpzTdRb5oXH6uwwAa0rYUGVejNpoZyg=s64", "userId": "06295554822278854914"}, "user_tz": 240} id="rJ9449R6yaPe" outputId="d2412c34-73a5-49e5-e92a-01bf091ee85e"
sent_test = sentences_high_syn
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = tt_notarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt2 = tt_wttarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt1 = (tt1-tt1.min())/(tt1.max()-tt1.min())    
    tt2 = (tt2-tt2.min())/(tt2.max()-tt2.min())    
    
    axes[i][0].bar(x=sent_test[1][i], height=tt1, width=0.2)
    axes[i][1].bar(x=sent_test[1][i], height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")    
```

<!-- #region {"colab_type": "text", "id": "mWvVHmCSzShW"} -->
## antonym
<!-- #endregion -->

```python
sent_test = sentences_high_ant
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = tt_notarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt2 = tt_wttarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt1 = (tt1-tt1.min())/(tt1.max()-tt1.min())    
    tt2 = (tt2-tt2.min())/(tt2.max()-tt2.min())    
    
    axes[i][0].bar(x=sent_test[1][i], height=tt1, width=0.2)
    axes[i][1].bar(x=sent_test[1][i], height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")    
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)    
```

<!-- #region {"colab_type": "text", "id": "pfnzJsSVzbyP"} -->
## causal
<!-- #endregion -->

```python
sent_test = sentences_high_cau
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = tt_notarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt2 = tt_wttarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt1 = (tt1-tt1.min())/(tt1.max()-tt1.min())    
    tt2 = (tt2-tt2.min())/(tt2.max()-tt2.min())    
    
    axes[i][0].bar(x=sent_test[1][i], height=tt1, width=0.2)
    axes[i][1].bar(x=sent_test[1][i], height=tt2, width=0.2)
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
sentences_hp
```

```python colab={} colab_type="code" id="wtE2eSxT9trv"
sentences_homo = [xx[df_ex_homopoly["type"]=="homonymy"] for xx in sentences_hp]
sentences_poly = [xx[df_ex_homopoly["type"]=="polysemy"] for xx in sentences_hp]
```

```python colab={} colab_type="code" id="wFmRkvNp9egL"

```

```python
sent_test = sentences_homo
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = tt_notarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt2 = tt_wttarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt1 = (tt1-tt1.min())/(tt1.max()-tt1.min())    
    tt2 = (tt2-tt2.min())/(tt2.max()-tt2.min())    
    
    axes[i][0].bar(x=sent_test[1][i], height=tt1, width=0.2)
    axes[i][1].bar(x=sent_test[1][i], height=tt2, width=0.2)
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
sent_test = sentences_poly
tt_notarg_attn = model_notarg_sfmxres.predict(sent_test)[:,0,:]
tt_wttarg_attn = model_wttarg_sfmxres.predict(sent_test)[:,0,:]

fig, axes = plt.subplots(len(tt_notarg_attn), 2, figsize=(16, len(tt_notarg_attn)*3))
fig.subplots_adjust(hspace=2)

for i in range(len(tt_wttarg_attn)):
    tt1 = tt_notarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt2 = tt_wttarg_attn[i]*(sent_test[2][i]+sent_test[3][i])
    tt1 = (tt1-tt1.min())/(tt1.max()-tt1.min())    
    tt2 = (tt2-tt2.min())/(tt2.max()-tt2.min())    
    
    axes[i][0].bar(x=sent_test[1][i], height=tt1, width=0.2)
    axes[i][1].bar(x=sent_test[1][i], height=tt2, width=0.2)
    axes[i][0].set_title("notarg ")
    axes[i][1].set_title("wttarg ")
    axes[i][0].tick_params(axis='x', rotation=90)
    axes[i][1].tick_params(axis='x', rotation=90)    
```

```python

```
