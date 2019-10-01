import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from models.build_models import build_model_elmo, build_model_bert, initialize_vars
import keras_tqdm
from livelossplot.keras import PlotLossesCallback

def train_elmomod_cv(sentences, resp_scores, kf_split, _att,
                     model_weight_loc, model_pred_loc,
                     max_seq_len, _num_iter, _batch_size):
    _fold_idx = 0
    plot_losses = PlotLossesCallback()
#     checkpoint = ModelCheckpoint(model_weight_loc+str(_fold_idx)+".h5", 
#                                  monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    
    for train_idx, test_idx in kf_split:
        K.clear_session()
        sess = tf.Session()

        print("fold:", _fold_idx)
        # preparing
        _mod = build_model_elmo(max_seq_len, attention_layer=_att)
        initialize_vars(sess)

        _sent_train = [sent[train_idx] for sent in sentences]
        _sent_test  = [sent[test_idx] for sent in sentences]
        _resp_train  = [resp_scores[i] for i in train_idx]
        _resp_test   = [resp_scores[i] for i in test_idx]

        # training
        _mod.fit(x=_sent_train, y=_resp_train, 
                 epochs=_num_iter, batch_size=_batch_size, 
                 validation_split=0.10, shuffle=True,
                 verbose=0, 
                 callbacks=[keras_tqdm.TQDMNotebookCallback(leave_inner=False, leave_outer=True), 
                            plot_losses])
        _mod.save_weights(model_weight_loc+str(_fold_idx)+".h5")

        # prediction
        _pred_test = np.reshape(_mod.predict(_sent_test, batch_size=_batch_size), -1)    
        np.save(model_pred_loc+str(_fold_idx)+".npy", _pred_test)

        _fold_idx += 1


def train_bertmod_cv(sentences, resp_scores, kf_split, _att,
                 model_weight_loc, model_pred_loc,
                 max_seq_len, _num_iter, _batch_size):
    _fold_idx = 0
    plot_losses = PlotLossesCallback()
    
    for train_idx, test_idx in kf_split:
        K.clear_session()
        sess = tf.Session()

        print("fold:", _fold_idx)
        # preparing
        _mod = build_model_elmo(max_seq_len, attention_layer=_att)
        initialize_vars(sess)

        _sent_train = [sent[train_idx] for sent in sentences]
        _sent_test  = [sent[test_idx] for sent in sentences]
        _resp_train  = [resp_scores[i] for i in train_idx]
        _resp_test   = [resp_scores[i] for i in test_idx]

        _train_examples = convert_text_to_examples(_sent_train[0], _sent_train[1], _resp_train[i])
        _test_examples  = convert_text_to_examples(_sent_test[0],  _sent_test[1],  _resp_test[i])
        (_train_input_ids, _train_input_masks, _train_segment_ids, _train_scores) = convert_examples_to_features(tokenizer, _train_examples, MAX_SEQ_LEN)
        (_test_input_ids,  _test_input_masks,  _test_segment_ids,  _test_scores)  = convert_examples_to_features(tokenizer, _test_examples,  MAX_SEQ_LEN)
        

    
        # training
        _mod.fit(x=[_train_input_ids[i], _train_input_masks[i], _train_segment_ids[i]], y=_resp_train[i], 
                 epochs=_num_iter, batch_size=_batch_size, 
                 validation_split=0.10, shuffle=True,
                 verbose=0, 
                 callbacks=[keras_tqdm.TQDMNotebookCallback(leave_inner=False, leave_outer=True), plot_losses])
        _mod.save_weights(model_weight_loc+str(_fold_idx)+".h5")

        # prediction
        _pred_test = np.reshape(_mod.predict([_test_input_ids, _test_input_masks, _test_segment_ids], 
                                             batch_size=_batch_size), -1)    
        np.save(model_pred_loc+str(_fold_idx)+".npy", _pred_test)

        _fold_idx += 1  
        
        
class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """    

    
class InputExample(object):
    # a single training/text example
    def __init__(self, guid, targ, text_a, text_b=None, score=None):
        self.guid = guid
        self.targ = targ
        self.text_a = text_a
        self.text_b = text_b
        self.score = score

def create_tokenizer_from_hub_module():
    # get the vocab file and casing info from the TfHub module
    bert_module = hub.Module(BERT_PATH)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        
        
def convert_text_to_examples(targets, sentences, scores):
    # create `InputExample`s
    InputExamples = []
    for targ, sent, score in zip(targets, sentences, scores):
        # InputExamples.append(InputExample(guid=None, text_a=" ".join(sent), text_b=None, score=score)) # for the array of multiple sentences as a single string
        InputExamples.append(InputExample(guid=None, targ=targ, text_a=sent, text_b=None, score=score))
    return InputExamples

def convert_single_example(tokenizer, example, max_seq_length=256):
    # converts a *single* `InputExample` into a single `InputFeatures`
    
    ## For TPU fixed sized paddings
    if isinstance(example, PaddingInputExample):
        input_ids = [0]*max_seq_length
        input_mask = [0]*max_seq_length
        segment_ids = [0]*max_seq_length
        score = 0
        return input_ids, input_mask, segment_ids, score
    
    ## else: inputs for regular GPU sessions
    tokens_targ = tokenizer.tokenize(example.targ)
    tokens_a = tokenizer.tokenize(example.text_a)
#     if len(tokens_a) > max_seq_length - 2:                 # -2 when [CLS] and [SEP] are included in the data
#         tokens_a = tokens_a[0: (max_seq_length - 2)]
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0: (max_seq_length)]
    
    tokens_sent = []
    segment_ids = []
#     tokens_sent.append("[CLS]") # useful for sentence classification task
#     segment_ids.append(0)       # segment_id for [CLS] token
    for token in tokens_a:
        tokens_sent.append(token)
        segment_ids.append(0)
#     tokens_sent.append("[SEP]") # useful for sentence classification & multisentence task
#     segment_ids.append(0)       # segment_id for [SEP] token
    
    targ_ids = tokenizer.convert_tokens_to_ids(tokens_targ)
    input_ids = tokenizer.convert_tokens_to_ids(tokens_sent)
        
    # mask: 1 for real tokens; 0 for padding tokens
    input_mask = [1]*len(input_ids)
    
    # update mask for target word tokens as 0
    targ_idx = [input_ids.index(x) for x in targ_ids]
    for i in targ_idx:
        input_mask[i] = 0
    
    # zero padding up to the sequence length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    return input_ids, input_mask, segment_ids, example.score


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    # converts a *set* of `InputExample`s to a list of `InputFeatures`
    input_ids, input_masks, segment_ids, scores = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, score = convert_single_example(tokenizer, example, max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        scores.append(score)
    
    return(np.array(input_ids), np.array(input_masks), np.array(segment_ids), np.array(scores).reshape(-1, 1))
        