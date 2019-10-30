# from keras import layers, Model, regularizers, optimizers
from tensorflow.keras import layers, Model, regularizers, optimizers
from tensorflow.keras import backend as K

from layers.embeddings import ElmoLayer, BertLayer
from layers.attention import AttentionLayer
import tensorflow as tf
import numpy as np

# multiplier function for masking
mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)

# global learning rate parameter
LEARNING_RATE = 3e-5    #keras default: 0.001

# ===== ELMo model =====
# references:
# https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
# https://github.com/AlexYangLi/ABSA_Keras/blob/master/custom_layers.py
# https://github.com/AlexYangLi/ABSA_Keras/blob/4392a52d08640b2b6c6aa586c4468f446fa6d2b7/models.py
def build_model_elmo(max_seq_len, finetune_emb, attention_layer, sep_cntx_targ=False, lr=LEARNING_RATE):
    # inputs
    input_sent_len = layers.Input((1,), name='input_sent_len', dtype="int32")
    input_sentence = layers.Input((max_seq_len,), name='input_sentence', dtype="string")
    input_mask = layers.Input((max_seq_len,), name='input_mask', dtype="int32")
    input_tloc = layers.Input((max_seq_len,), name='input_tloc', dtype="int32") 
    
    # fetch ELMo embeddings
    elmo_cntx, elmo_targ = elmo_embed(input_sent_len, input_sentence, input_mask, input_tloc, sep_cntx_targ=sep_cntx_targ)

    cntx_targ = None
    if(attention_layer):
        ## context words to the target word
        # cntx_targ = attention_block_mul(elmo_cntx, elmo_targ, input_sent_len, max_seq_len)
        cntx_targ = attention_block_add(elmo_cntx, elmo_targ, max_seq_len)
    else:
        cntx_targ = elmo_targ # updated target word (e.g., <UNK>) from ELMo's bidirectional process

    # final regression output MLP
    output = regression_block(cntx_targ)

    # model compile
    model = Model([input_sent_len, input_sentence, input_mask, input_tloc], output)   # adam = optimizers.Adam(lr=0.0001)
    
    # Keras bug: manually setting the trainability of embedding layer(s)
    if(not finetune_emb):
        if(sep_cntx_targ):
            model.get_layer("elmo_cntx_raw").trainable=False
            model.get_layer("elmo_targ_raw").trainable=False
        else:
            model.get_layer("elmo_sent_raw").trainable=False

    adam = optimizers.Adam(lr=lr)
    model.compile(loss='mean_absolute_error', optimizer=adam)
#     model.compile(loss='mean_absolute_error', optimizer="adam")


    return model


def build_model_elmo_kapelner(max_seq_len, attention_layer):
    # inputs
    input_sent_len = layers.Input((1,), name='input_sent_len', dtype="int32")
    input_sentence = layers.Input((max_seq_len,), name='input_sentence', dtype="string")
    input_mask = layers.Input((max_seq_len,), name='input_mask', dtype="int32")
    input_tloc = layers.Input((max_seq_len,), name='input_tloc', dtype="int32") 
    
    input_lexf = layers.Input((30,), name="input_lexf", dtype="float32")
    
    elmo_input = [input_sent_len, input_sentence]
    # fetch ELMo embeddings
    elmo_sent = ElmoLayer(trainable=True, name="elmo_sent")(elmo_input)
    elmo_cntx = layers.Lambda(lambda x: mul_mask(x[0], tf.cast(x[1], tf.float32)), name="elmo_cntx")([elmo_sent, input_mask])
    elmo_targ = layers.Lambda(lambda x: tf.reduce_sum(mul_mask(x[0], tf.cast(x[1], tf.float32)), axis=1), name="elmo_targ")([elmo_sent, input_tloc])

    cntx_targ = None
    if(attention_layer):
        ## context words to the target word
        tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(elmo_targ)
        attn_layer = AttentionLayer(name='attention_layer')        
        att_out, attn_states = attn_layer([elmo_cntx, tg_emb_repeat], verbose=False)
        
        att_out = layers.GlobalAveragePooling1D()(att_out)
        cntx_targ = att_out
    else:
        cntx_targ = elmo_targ # updated target word (e.g., <UNK>) from ELMo's bidirectional process

    # final regression output MLP
    cntx_targ_feat = layers.concatenate([cntx_targ, input_lexf])
    output = layers.Dense(256, activation='relu')(cntx_targ_feat)
    output = layers.Dense(1, activation='linear')(output) #, kernel_regularizer=regularizers.l2(np.exp(0.1)))(mem)

    # model compile
    model = Model([input_sent_len, input_sentence, input_mask, input_tloc, input_lexf], output)
    # adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model


# ===== BERT model =====
# reference: https://github.com/strongio/keras-bert/blob/master/keras-bert.ipynb
# enabling the finetuning hurts the performance of dscovar sentences (e.g., rocauc .78 vs .5)
def build_model_bert(max_seq_len, finetune_emb, attention_layer, sep_cntx_targ=False, lr=LEARNING_RATE):
    input_id = layers.Input(shape=(max_seq_len), name="input_ids")
    input_mask = layers.Input(shape=(max_seq_len), name="input_masks")
    input_segment = layers.Input(shape=(max_seq_len), name="segment_ids")
    input_tloc = layers.Input(shape=(max_seq_len), name="input_tloc")    
    
    bert_cntx, bert_targ = bert_embed(input_id, input_mask, input_segment, input_tloc, sep_cntx_targ=sep_cntx_targ)

    cntx_targ = None
    if(attention_layer):
        ## context words to the target word
        cntx_targ = attention_block_add(bert_cntx, bert_targ, max_seq_len)
    else:
        cntx_targ = bert_targ # updated target word (e.g., <UNK>) from BERT's process
    
    # final regression output MLP
    output = regression_block(cntx_targ)

    # model compile
    model = Model([input_id, input_mask, input_segment, input_tloc], output)

    # Keras bug: manually setting the trainability of embedding layer(s)
    if(not finetune_emb):
        if(sep_cntx_targ):
            model.get_layer("bert_cntx_raw").trainable=False
            model.get_layer("bert_targ_raw").trainable=False
        else:
            model.get_layer("bert_sent_raw").trainable=False
    
    adam = optimizers.Adam(lr=lr)
    model.compile(loss='mean_absolute_error', optimizer=adam)

    return model


def elmo_embed(input_sent_len, input_sentence, input_mask, input_tloc, sep_cntx_targ=False):   
    elmo_cntx = None
    elmo_targ = None
    if(sep_cntx_targ):
        elmo_cntx = ElmoLayer(name="elmo_cntx_raw")([input_sent_len, input_sentence])
        elmo_targ = ElmoLayer(name="elmo_targ_raw")([input_sent_len, input_sentence])
        elmo_cntx = layers.Lambda(lambda x:               mul_mask(x[0], tf.cast(x[1], tf.float32)),          name="elmo_cntx")([elmo_cntx, input_mask])
        elmo_targ = layers.Lambda(lambda x: tf.reduce_sum(mul_mask(x[0], tf.cast(x[1], tf.float32)), axis=1), name="elmo_targ")([elmo_targ, input_tloc]) # sum with 0s -> make it as a single vector
#         elmo_targ = layers.Lambda(lambda x: tf.reduce_sum(mul_mask(x[0], tf.cast(tf.equal(x[1], 0), tf.float32)), axis=1), name="elmo_targ")([elmo_targ, input_tloc])
    else:
        elmo_sent = ElmoLayer(name="elmo_sent_raw")([input_sent_len, input_sentence])
        elmo_cntx = layers.Lambda(lambda x:               mul_mask(x[0], tf.cast(x[1], tf.float32)),          name="elmo_cntx")([elmo_sent, input_mask])
        elmo_targ = layers.Lambda(lambda x: tf.reduce_sum(mul_mask(x[0], tf.cast(x[1], tf.float32)), axis=1), name="elmo_targ")([elmo_sent, input_tloc])
    return(elmo_cntx, elmo_targ)


def bert_embed(input_id, input_mask, input_segment, input_tloc, sep_cntx_targ=False):
    bert_cntx = None
    bert_targ = None
    if(sep_cntx_targ):
        bert_cntx = BertLayer(pooling="seq", name="bert_cntx_raw")([input_id, input_mask, input_segment])
        bert_targ = BertLayer(pooling="seq", name="bert_targ_raw")([input_id, input_mask, input_segment])        
        bert_cntx = layers.Lambda(lambda x:               mul_mask(x[0], tf.cast(x[1], tf.float32)),          name="bert_cntx")([bert_cntx, input_mask])
        bert_targ = layers.Lambda(lambda x: tf.reduce_sum(mul_mask(x[0], tf.cast(x[1], tf.float32)), axis=1), name="bert_targ")([bert_targ, input_tloc])
    else:
        bert_sent = BertLayer(pooling="seq", name="bert_sent_raw")([input_id, input_mask, input_segment])
        bert_cntx = layers.Lambda(lambda x:               mul_mask(x[0], tf.cast(x[1], tf.float32)),          name="bert_cntx")([bert_sent, input_mask])
        bert_targ = layers.Lambda(lambda x: tf.reduce_sum(mul_mask(x[0], tf.cast(x[1], tf.float32)), axis=1), name="bert_targ")([bert_sent, input_tloc])
    return(bert_cntx, bert_targ)
        

def attention_block_add(in_vec_cntx, in_vec_targ, max_seq_len):
    tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(in_vec_targ)
    attn_layer = AttentionLayer(name='attention_layer')        
    att_out, attn_states = attn_layer([in_vec_cntx, tg_emb_repeat], verbose=False)
        
    att_out = layers.GlobalAveragePooling1D()(att_out)
    return(att_out)

def attention_block_mul(in_vec_cntx, in_vec_targ, sent_len, max_seq_len):
    tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(in_vec_targ)
    att_fout = layers.Attention(name="attention_fout")([in_vec_cntx, in_vec_targ])
    # att_fout = layers.Dense(1024)(att_fout)
    # att_fout = att_fout(inputs=[in_vec_cntx, in_vec_targ], mask=[_mask,None])

    # TODO: insert mask here
    # TODO: no weights learned in tf.keras built in attention layers?
    # sent_mask = layers.Lambda(lambda x:tf.concat([tf.ones(x[0]), tf.zeros(x[1]-x[0])], axis=0))([sent_len, max_seq_len])
    # sent_mask = attn_mask
    att_fout = layers.Lambda(lambda x:mul_mask(x[0], tf.cast(tf.range(max_seq_len)<x[1], tf.float32)), name="attention_fout_masked")([att_fout, sent_len])

    # att_sfmx = layers.Dense(activation='softmax', name="attention_sfmx")(att_fout)
    att_sfmx = layers.Dense(30, use_bias=False, activation='softmax', name="attention_sfmx")(att_fout)
    # att_out = layers.Lambda(lambda x:tf.matmul(x[0], x[1]), name="attention_wvec")([att_fout, att_sfmx])
    att_out = layers.multiply([att_fout, tf.transpose(att_sfmx)], name="attention_wvec")
    att_out = layers.GlobalAveragePooling1D(name="attention_rout")(att_out)
    return(att_out)
    

def regression_block(in_vec):
    reg_out = layers.Dense(256, activation='relu', name='reg_rectify')(in_vec)
    reg_out = layers.Dense(1, activation='linear', name='reg_score_out')(reg_out) #, kernel_regularizer=regularizers.l2(np.exp(0.1)))(mem)
    return(reg_out)



def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)
    
    
    





#     if(attention_layer):
#         att_targ = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh'), name="attention_tanh_targ")(mem_targ)
#         att_targ = tf.keras.layers.Softmax(name="attention_softmax_targ", axis=1)(att_targ)
# #         att_to_targ = tf.keras.layers.Lambda(lambda x: x[0]*tf.expand_dims(tf.cast(x[1], tf.float32), axis=-1), 
# #                                              name="attention_masked")([att_to_targ, in_mask])  # masking & normalizing not helpful?
#         mem_targ = tf.keras.layers.multiply([mem_targ, att_targ])    



# def build_model_elmo(max_seq_len, attention_layer):
#     # inputs
#     input_sent_len = layers.Input((1,), name='input_sent_len', dtype="int32")
#     input_sentence = layers.Input((max_seq_len,), name='input_sentence', dtype="string")
# #     input_targ_idx = layers.Input((1,), name='input_targ_idx', dtype="int32")
#     input_mask_LH  = layers.Input((max_seq_len,), name='input_mask_LH', dtype="int32")
#     input_mask_RH  = layers.Input((max_seq_len,), name='input_mask_RH', dtype="int32")

#     # fetch ELMo embeddings
#     ## context words L/R to the target word
#     elmo_LH = ElmoLayer(side="LH", name="elmo_cntx_LH")  # keras having name reference problem for custom layers? e.g., if the name is assigned for ELMo embedding layer, the later models refer to the first model's elmo layer...?
#     elmo_RH = ElmoLayer(side="RH", name="elmo_cntx_RH")
#     elmo_tg = ElmoLayer(side="tg", name="elmo_targ")

#     elmo_LH_emb = elmo_LH([input_sent_len, input_sentence, input_mask_LH, input_mask_RH])
#     elmo_RH_emb = elmo_RH([input_sent_len, input_sentence, input_mask_LH, input_mask_RH])
#     elmo_tg_emb = elmo_tg([input_sent_len, input_sentence, input_mask_LH, input_mask_RH])
# #     print(elmo_RH_emb, elmo_tg_emb)
    
#     # using sequenced ELMo embeddings as memory
#     ## why ADD; not concatenate? -> fills LH 0s with RH values (and vice versa)
#     # mem = layers.Add(name="cntx_to_targ")([elmo_LH_emb, elmo_RH_emb])
    
#     cntx_targ = None
#     if(attention_layer):
#         ## context words to the target word
#         mem = layers.Add(name="cntx_to_targ")([elmo_LH_emb, elmo_RH_emb])        
#         tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(elmo_tg_emb)
#         attn_layer = AttentionLayer(name='attention_layer')        
#         att_out, attn_states = attn_layer([mem, tg_emb_repeat], verbose=False)
        
#         att_out = layers.GlobalAveragePooling1D()(att_out)
# #         mem = layers.GlobalAveragePooling1D()(mem)      
#         cntx_targ = att_out
# #         mem = layers.Concatenate(name='concat_att_mem')([mem, att_out])
# #         attn_states = layers.Attention(name='attention_layer', use_scale=False)([mem_cntx, mem_cntx])
# #         attn_states = layers.Softmax(axis=1)(attn_states)
# #         print(attn_states)
# #         mem_cntx = layers.multiply([mem_cntx, attn_states])
#         # mem_cntx = layers.Concatenate(axis=-1, name='concat_att_mem')([mem_cntx, att_out])
#     else:
#         # mem = layers.GlobalAveragePooling1D()(mem)
#         cntx_targ = elmo_tg_emb # updated target word (e.g., <UNK>) from ELMo's bidirectional process

#     # mem = layers.Add(name='cntx_and_targ')([mem_cntx, elmo_LH_emb, elmo_tg_emb, elmo_RH_emb])
#     # mem = layers.GlobalAveragePooling1D()(mem_cntx)

#     # final regression output MLP
# #     output = layers.Dense(512, activation='relu')(mem)
# #     # output = layers.Activation('relu')(mem)
# #     output = tf.keras.layers.Dense(512, activation='linear', kernel_initializer='normal')(output)
# #     output = layers.Dropout(0.2)(output)
#     output = layers.Dense(256, activation='relu')(cntx_targ)
#     # output = layers.Activation('relu')(output)
#     output = layers.Dense(1, activation='linear')(output) #, kernel_regularizer=regularizers.l2(np.exp(0.1)))(mem)

#     # model compile
#     model = Model([input_sent_len, input_sentence, input_mask_LH, input_mask_RH], output)
#     # adam = optimizers.Adam(lr=0.0001)
#     model.compile(loss='mean_absolute_error', optimizer='adam')

#     return model


# def build_model_bert(max_seq_len, attention_layer):
#     in_id = layers.Input(shape=(max_seq_len), name="input_ids")
#     in_mask = layers.Input(shape=(max_seq_len), name="input_masks")
#     in_tloc = layers.Input(shape=(max_seq_len), name="input_tloc")    
#     in_segment = layers.Input(shape=(max_seq_len), name="segment_ids")
#     bert_inputs = [in_id, in_mask, in_tloc, in_segment]
    
#     bert_cntx = BertLayer(pooling="mean", name="bert_cntx")(bert_inputs)
#     bert_targ = BertLayer(pooling="targ", name="bert_targ")(bert_inputs)

#     cntx_targ = None
#     if(attention_layer):
#         ## context words to the target word
#         tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(bert_targ)
#         attn_layer = AttentionLayer(name='attention_layer')        
#         att_out, attn_states = attn_layer([bert_cntx, tg_emb_repeat], verbose=False)
        
#         att_out = layers.GlobalAveragePooling1D()(att_out)
#         cntx_targ = att_out
#     else:
#         cntx_targ = bert_targ # updated target word (e.g., <UNK>) from BERT's process
    
#     # final regression output MLP
#     output = layers.Dense(256, activation='relu')(cntx_targ)
#     output = layers.Dense(1, activation='linear')(output) #, kernel_regularizer=regularizers.l2(np.exp(0.1)))(mem)

#     # model compile
#     model = Model([in_id, in_mask, in_tloc, in_segment], output)
#     # adam = optimizers.Adam(lr=0.0001)
#     model.compile(loss='mean_absolute_error', optimizer='adam')

#     return model


# def build_model_bert(max_seq_len, attention_layer):
#     in_id = layers.Input(shape=(max_seq_len), name="input_ids")
#     in_mask = layers.Input(shape=(max_seq_len), name="input_masks")
#     in_tloc = layers.Input(shape=(max_seq_len), name="input_tloc")    
#     in_segment = layers.Input(shape=(max_seq_len), name="segment_ids")
#     bert_inputs = [in_id, in_mask, in_segment]
    
#     bert_sent = BertLayer(trainable=False, pooling="seq", name="bert_sent")(bert_inputs)
#     bert_cntx = layers.Lambda(lambda x: mul_mask(x[0], tf.cast(x[1], tf.float32)), name="bert_cntx")([bert_sent, in_mask])
#     bert_targ = layers.Lambda(lambda x: tf.reduce_sum(mul_mask(x[0], tf.cast(x[1], tf.float32)), axis=1)/ \
#                                         tf.reduce_sum(x[1], axis=1, keepdims=True), 
#                               name="bert_targ")([bert_sent, in_tloc])

#     cntx_targ = None
#     if(attention_layer):
#         ## context words to the target word
#         tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(bert_targ)
#         attn_layer = AttentionLayer(name='attention_layer')        
#         att_out, attn_states = attn_layer([bert_cntx, tg_emb_repeat], verbose=False)
        
#         att_out = layers.GlobalAveragePooling1D()(att_out)
#         cntx_targ = att_out
#     else:
#         cntx_targ = bert_targ # updated target word (e.g., <UNK>) from BERT's process
    
#     # final regression output MLP
#     output = layers.Dense(256, activation='relu')(cntx_targ)
#     output = layers.Dense(1, activation='linear')(output) #, kernel_regularizer=regularizers.l2(np.exp(0.1)))(mem)

#     # model compile
#     model = Model([in_id, in_mask, in_tloc, in_segment], output)
#     # adam = optimizers.Adam(lr=0.0001)
#     model.compile(loss='mean_absolute_error', optimizer='adam')

#     return model