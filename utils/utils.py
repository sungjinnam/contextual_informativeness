import numpy as np
import glob
from sklearn.metrics import roc_curve, auc
from scipy import interp
from scipy import stats
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

FILTERS = '"#$%&()*+/:;=@[\\]^_`{|}~\t\n'
RELATIONAL_CUE_TOKENS = {
    'IsA': ['kind'], 
    'Antonym': ['opposite'], 
    'Synonym': ['same'], 
    'PartOf': ['part'], 
    'MemberOf': ['member'], 
    'MadeOf': ['made'], 
    'Entails': ['true'], #['also', 'true'], 
    'HasA': ['have', 'contain'], 
    'HasProperty': ['characterized']  #'specify' - typo in the paper?
}

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.array(y_pred) - np.array(y_true)), axis=-1))

def rocauc(y_true, y_pred, cut, direction="high"):
    if(direction=="high"):
        fpr, tpr, _ = roc_curve(y_true > np.quantile(y_true, q=[cut]), y_pred)
    if(direction=="low"):
        fpr, tpr, _ = roc_curve(y_true < np.quantile(y_true, q=[cut]), y_pred)

    roc_auc = auc(fpr, tpr)
    return (roc_auc)

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


def proc_sentences_dscovar(df, col_sentence, col_targ, mod_type, max_seq_len):
    if(mod_type=='bert'):
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
    
    elif(mod_type=='elmo'):
        sentences = []
        li_mask_cntx = []
        li_mask_targ = []
        li_sent_len = []
        li_sent_pad = []
        for i in range(df.shape[0]):
            sent = df.iloc[i][col_sentence]

            targ = None
            if(col_targ):
                targ = df.iloc[i][col_targ]
            else:
                targ = "<UNK>"

            sent = sent.replace("______", targ)
            sent = sent.replace("<BOS>", "").replace(".", " .").replace(",", " ,").replace("!", " !").replace(",?", " ?").replace("'s", " 's")
            sent_tok = text_to_word_sequence(sent, lower=False, filters=FILTERS)
            sent_pad = pad_sequences([sent_tok], maxlen=max_seq_len, dtype='object', padding='post', value=[""])       
            targ_idx = np.where(targ==sent_pad[0])[0][0]

            mask_targ = [0]*(max_seq_len)
            mask_targ[targ_idx] = 1
            mask_cntx = [0]*(max_seq_len)
            for i in range(targ_idx):
                mask_cntx[i] = 1
            if(col_targ):
                for i in range(targ_idx, len(sent_tok)):
                    mask_cntx[i] = 1
            else:
                for i in range(targ_idx+1, len(sent_tok)):
                    mask_cntx[i] = 1

            sent_len = len(sent_tok)

            li_sent_len.append(sent_len)
            li_sent_pad.append(list(sent_pad)[0])
            li_mask_cntx.append(mask_cntx)
            li_mask_targ.append(mask_targ)
        sentences = [np.array(li_sent_len), np.array(li_sent_pad), np.array(li_mask_cntx), np.array(li_mask_targ)]
        return(sentences)

    
def proc_sentences_EVALution(df, col_sentence, col_targ, col_targpair, col_relation, 
                             include_targ, max_seq_len, rel_tokens):
    sentences = []
    li_mask_cntx = []
    li_mask_targ = []
    li_mask_rcue  = []
    li_mask_pair = []
    
    li_sent_len = []
    li_sent_pad = []
    for k in range(df.shape[0]):
        sent = df.iloc[k][col_sentence]
        targ = df.iloc[k][col_targ]
        targ_pair = df.iloc[k][col_targpair]
        rel_type = df.iloc[k][col_relation]
            
        sent = sent.replace("<BOS>", "").replace(".", " .").replace(",", " ,").replace("!", " !").replace(",?", " ?").replace("'s", " 's")
        sent_tok = text_to_word_sequence(sent, lower=False, filters=FILTERS)
        sent_pad = pad_sequences([sent_tok], maxlen=max_seq_len, dtype='object', padding='post', value=[""])
  
        targ_idx = matching_tok_idx(sent_pad[0], targ.split('_'))
        rcue_idx = matching_tok_idx(sent_pad[0], rel_tokens[rel_type])
        pair_idx = matching_tok_idx(sent_pad[0], targ_pair.split('_'))
        # print(sent_pad, rel_tokens[rel_type])

        # target word masking
        mask_targ = [0]*(max_seq_len)
        for i in targ_idx:
            mask_targ[i] = 1
        
        # context masking
        mask_cntx = [0]*(max_seq_len)
        # print(k, targ, targ_idx, rel_type, rel_tokens[rel_type], sent_pad)
        ## fill the LH context
        for i in range(targ_idx[0]): 
            mask_cntx[i] = 1
            
        ## fill the RH context
        if(include_targ):
            for i in range(targ_idx[-1], len(sent_tok)):
                mask_cntx[i] = 1
        else:
            for i in range(targ_idx[-1]+1, len(sent_tok)):
                mask_cntx[i] = 1
            for i in targ_idx:
                sent_pad[0][i] = "<UNK>"
        
        # target word pair masking
        mask_pair = [0]*(max_seq_len)
        for i in pair_idx:
            mask_pair[i] = 1
        
        # relational cue masking
        mask_rcue = [0]*(max_seq_len)
        for i in rcue_idx:
            mask_rcue[i] = 1
            
        sent_len = len(sent_tok)
              
        li_sent_len.append(sent_len)
        li_sent_pad.append(list(sent_pad)[0])
        li_mask_cntx.append(mask_cntx)
        li_mask_targ.append(mask_targ)
        li_mask_pair.append(mask_pair)
        li_mask_rcue.append(mask_rcue)
    sentences = [np.array(li_sent_len), np.array(li_sent_pad), 
                 np.array(li_mask_cntx), np.array(li_mask_targ), 
                 np.array(li_mask_pair), np.array(li_mask_rcue)]
    return(sentences)

def matching_tok_idx(sent_tok, comp_tok):
    ret = []
    for x in comp_tok:
        for i,tok in enumerate(sent_tok):
            if(x.lower()==tok.lower()):
                ret.append(i)
    return(ret)

def split_and_lower(x):
    return([xx.lower() for xx in x.split('_')])

# def attention_ranks_dummy(sent_len, comp_mask):
    

def attention_ranks(_attn_weights, sent_len, comp_mask):
    attn_weights = _attn_weights[:sent_len]
    attn_weights_rank = np.argsort(attn_weights)[::-1]
    comp_idx = np.where(comp_mask)[0]
    ret = np.array([np.where(x==attn_weights_rank)[0][0] for x in comp_idx])
    return(ret)

def attention_scores(attn_weights, sent_len, pair_mask, rcue_mask):
    rank_pair = attention_ranks(attn_weights, sent_len, pair_mask)
    rank_rcue = attention_ranks(attn_weights, sent_len, rcue_mask)
    score_pair = ((sent_len-rank_pair)/sent_len).mean()
    score_rcue = ((sent_len-rank_rcue)/sent_len).mean()
    return(np.array([score_pair, score_rcue]))


def min_max_sc(_x): 
    x = np.array(_x)
    return (x-x.min())/(x.max()-x.min())

def ci(vec, confidence=0.95, digits=None):
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(vec)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    if(digits):
        return round(m, digits), round(m-h, digits), round(m+h, digits)
    else:
        return m, m-h, m+h


# # MAX_SEQ_LEN = 30
# def proc_sentences_elmo(df, col_sentence, col_targ):
#     sentences = []
# #     li_mask_LH = []
# #     li_mask_RH = []
#     li_mask_cntx = []
#     li_mask_targ = []
# #     li_targ = []
# #     li_targ_idx = []
#     li_sent_len = []
#     li_sent_pad = []
#     for i in range(df.shape[0]):
#         sent = df.iloc[i][col_sentence]
        
#         targ = None
#         if(col_targ):
#             targ = df.iloc[i][col_targ]
#         else:
#             targ = "<UNK>"
        
#         sent = sent.replace("______", targ)
#         sent = sent.replace("<BOS>", "").replace(".", " .").replace(",", " ,").replace("!", " !").replace(",?", " ?").replace("'s", " 's")
#         sent_tok = text_to_word_sequence(sent, lower=False, filters=FILTERS)
#         sent_pad = pad_sequences([sent_tok], maxlen=MAX_SEQ_LEN, dtype='object', padding='post', value=[""])       
#         targ_idx = np.where(targ==sent_pad[0])[0][0]
        
#         mask_targ = [0]*(MAX_SEQ_LEN)
#         mask_targ[targ_idx] = 1
# #         mask_LH = [0]*(MAX_SEQ_LEN)
# #         mask_RH = [0]*(MAX_SEQ_LEN)
#         mask_cntx = [0]*(MAX_SEQ_LEN)
#         for i in range(targ_idx):
#             mask_cntx[i] = 1
#         if(col_targ):
#             for i in range(targ_idx, len(sent_tok)):
#                 mask_cntx[i] = 1
#         else:
#             for i in range(targ_idx+1, len(sent_tok)):
#                 mask_cntx[i] = 1
    
# #         for i in range(targ_idx):
# #             mask_LH[i] = 1
# #         for i in range(targ_idx+1, len(sent_tok)):
# #             mask_RH[i] = 1
        
#         sent_len = len(sent_tok)
        
# #         li_targ.append(targ)
# #         li_targ_idx.append(targ_idx)
#         li_sent_len.append(sent_len)
#         li_sent_pad.append(list(sent_pad)[0])
#         li_mask_cntx.append(mask_cntx)
#         li_mask_targ.append(mask_targ)
# #         li_mask_LH.append(mask_LH)
# #         li_mask_RH.append(mask_RH)
# #     sentences = [np.array(li_targ_idx), np.array(li_sent_len), np.array(li_sent_pad), np.array(li_targ)]
# #     sentences = [np.array(li_targ_idx), np.array(li_sent_len), np.array(li_sent_pad)]
# #     sentences = [np.array(li_sent_len), np.array(li_sent_pad), # np.array(li_targ_idx),
# #                  np.array(li_mask_LH), np.array(li_mask_RH)] #, np.sum((li_mask_LH, li_mask_RH), axis=0)]
#     sentences = [np.array(li_sent_len), np.array(li_sent_pad), np.array(li_mask_cntx), np.array(li_mask_targ)]
#     return(sentences)