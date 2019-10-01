# from keras import layers, Model, regularizers, optimizers
from tensorflow.keras import layers, Model, regularizers, optimizers
from tensorflow.keras import backend as K

from layers.embeddings import ElmoLayer, BertLayer
# from layers.attention import AttentionLayer
import tensorflow as tf
import numpy as np

# ===== ELMo model =====
# references:
# https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
# https://github.com/AlexYangLi/ABSA_Keras/blob/master/custom_layers.py
# https://github.com/AlexYangLi/ABSA_Keras/blob/4392a52d08640b2b6c6aa586c4468f446fa6d2b7/models.py
def build_model_elmo(max_seq_len, attention_layer):
    # inputs
    input_sent_len = layers.Input((1,), name='input_sent_len', dtype="int32")
    input_sentence = layers.Input((max_seq_len,), name='input_sentence', dtype="string")
#     input_targ_idx = layers.Input((1,), name='input_targ_idx', dtype="int32")
    input_mask_LH  = layers.Input((max_seq_len,), name='input_mask_LH', dtype="int32")
    input_mask_RH  = layers.Input((max_seq_len,), name='input_mask_RH', dtype="int32")

    # fetch ELMo embeddings
    ## context words L/R to the target word
    elmo_LH = ElmoLayer(side="LH", name="elmo_cntx_LH")  # keras having name reference problem for custom layers? e.g., if the name is assigned for ELMo embedding layer, the later models refer to the first model's elmo layer...?
    elmo_RH = ElmoLayer(side="RH", name="elmo_cntx_RH")
    elmo_tg = ElmoLayer(side="tg", name="elmo_targ")

    elmo_LH_emb = elmo_LH([input_sent_len, input_sentence, input_mask_LH, input_mask_RH])
    elmo_RH_emb = elmo_RH([input_sent_len, input_sentence, input_mask_LH, input_mask_RH])
    elmo_tg_emb = elmo_tg([input_sent_len, input_sentence, input_mask_LH, input_mask_RH])
#     print(elmo_RH_emb, elmo_tg_emb)
    
    # using sequenced ELMo embeddings as memory
    ## why ADD; not concatenate? -> fills LH 0s with RH values (and vice versa)
    mem_cntx = layers.Add(name="cntx_to_targ")([elmo_LH_emb, elmo_RH_emb])

    if(attention_layer):
        ## context words to the target word
        tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(elmo_tg_emb)
        # attn_layer = AttentionLayer(name='attention_layer')        
        # att_out, attn_states = attn_layer([mem_cntx, tg_emb_repeat], verbose=False)
        # mem_cntx = att_out
        attn_states = layers.Attention(name='attention_layer', use_scale=True)
        mem_cntx = layers.multiply([mem_cntx, attn_states])
        # mem_cntx = layers.Concatenate(axis=-1, name='concat_att_mem')([mem_cntx, att_out])

    # mem = layers.Add(name='cntx_and_targ')([mem_cntx, elmo_LH_emb, elmo_tg_emb, elmo_RH_emb])
    mem = layers.GlobalAveragePooling1D()(mem_cntx)

    # final regression output MLP
    output = layers.Dense(512, activation='relu')(mem)
    # output = layers.Activation('relu')(mem)
    output = tf.keras.layers.Dense(512, activation='linear', kernel_initializer='normal')(output)
    output = layers.Dropout(0.2)(output)
    output = layers.Dense(256, activation='relu')(output)
    # output = layers.Activation('relu')(output)
    output = layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(np.exp(1.0)))(output)

    # model compile
    model = Model([input_sent_len, input_sentence, input_mask_LH, input_mask_RH], output)
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model


# ===== BERT model =====
# reference: https://github.com/strongio/keras-bert/blob/master/keras-bert.ipynb
def build_model_bert(max_seq_length, attention_layer):
    in_id = tf.keras.layers.Input(shape=(max_seq_length), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]
    
    mem_targ = BertLayer(pooling="mean", name="bert_layer")(bert_inputs)
#     dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
#     pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    
#     model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()

    if(attention_layer):
        att_targ = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh'), name="attention_tanh_targ")(mem_targ)
        att_targ = tf.keras.layers.Softmax(name="attention_softmax_targ", axis=1)(att_targ)
#         att_to_targ = tf.keras.layers.Lambda(lambda x: x[0]*tf.expand_dims(tf.cast(x[1], tf.float32), axis=-1), 
#                                              name="attention_masked")([att_to_targ, in_mask])  # masking & normalizing not helpful?
        mem_targ = tf.keras.layers.multiply([mem_targ, att_targ])

    
    # bert time sequence pool
    mem_targ = tf.keras.layers.GlobalAveragePooling1D()(mem_targ)
    
    # final regression output MLP
    output = tf.keras.layers.Dense(512, activation='relu')(mem_targ)
    output = tf.keras.layers.Dense(512, activation='linear', kernel_initializer='normal')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(256, activation='relu')(output)
    output = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(np.exp(1.0)))(output)

    # model compile
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mean_absolute_error', optimizer=adam)
#     model.summary()
    
    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)