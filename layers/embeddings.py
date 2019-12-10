import tensorflow as tf
import tensorflow_hub as hub
# import keras.backend as K
# from keras.engine import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# TODO: reduce the number of module calling: minimize

# ===== ELMo layers using tf.hub =====
# references:
# https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
# https://github.com/AlexYangLi/ABSA_Keras/blob/master/custom_layers.py
# https://github.com/AlexYangLi/ABSA_Keras/blob/4392a52d08640b2b6c6aa586c4468f446fa6d2b7/models.py
ELMO_PATH = 'https://tfhub.dev/google/elmo/2'
BERT_PATH = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"
# BERT_PATH = "https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1"

class ElmoLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True   # trainability will be controlled in the model building process
        super(ElmoLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.elmo = hub.Module(ELMO_PATH, trainable=self.trainable, name="{}_module".format(self.name))
        if(self.trainable):            
            self._trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name)) # finetuning aggregation weights only (4 parameters)
            # self._trainable_weights += [var for var in self.elmo.variables] # finetuning all weights (180M parameters)
        super(ElmoLayer, self).build(input_shape)
        
    def call(self, inputs):
        input_len, input_tok = inputs
        input_len = K.squeeze(K.cast(input_len, tf.int32), axis=1)
        input_tok = input_tok

        result_raw = self.elmo(
            inputs={
                "tokens": input_tok,
                "sequence_len": input_len
            },
            signature="tokens",
            as_dict=True)

        result = result_raw['elmo']
        return result
    
    def compute_output_shape(self, input_shape):
        if(self.side == 'tg'):
            return (input_shape[0][0], int(self.dimensions))
        else:
            return (input_shape[0][0], input_shape[1][1], int(self.dimensions))
        

# ===== BERT layers using tf.hub =====
# reference: https://github.com/strongio/keras-bert/blob/master/keras-bert.ipynb
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=3, pooling='first', bert_path=BERT_PATH, **kwargs,):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True   # trainability will be controlled in the model building process
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        
        if self.pooling not in ["first", "seq"]:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling})")
        super(BertLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path, trainable=self.trainable, name=f"{self.name}_module")
        
        if(self.trainable):
            trainable_vars = self.bert.variables
            if self.pooling == "first":
                self._trainable_weights += [var for var in self.bert.variables if not "/cls/" in var.name]
            elif self.pooling == "seq":
#                 self._trainable_weights += [var for var in self.bert.variables if not "/cls/" in var.name and not "/pooler/" in var.name]
                trainable_vars = [var for var in self.bert.variables if not "/cls/" in var.name and not "/pooler/" in var.name]
                trainable_layers = []
            
            for i in range(self.n_fine_tune_layers):
                trainable_layers.append(f"encoder/layer_{str(11 - i)}")                
            trainable_vars = [var for var in trainable_vars if any([l in var.name for l in trainable_layers])]

            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
                
        super(BertLayer, self).build(input_shape)
        
    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)

        if self.pooling=="first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
        elif self.pooling == "seq":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
        return pooled
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0], self,output_size)
    
    
    
    
    
    
    
    
    
    
    
    
    
# class ElmoLayer(Layer):
#     def __init__(self, side, **kwargs):
#         self.dimensions = 1024
#         self.trainable = True
#         # self.direction = direction
#         self.side = side
#         # self.mask_zero = mask_zero
#         super(ElmoLayer, self).__init__(**kwargs)
        
#     def build(self, input_shape):        
#         self.elmo = hub.Module(ELMO_PATH, trainable=self.trainable, name="{}_module".format(self.name))
#         if(self.trainable):
#             self._trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoLayer, self).build(input_shape)
        
#     def call(self, inputs):
#         input_len, input_tok, input_mask_LH, input_mask_RH = inputs
#         input_len = K.squeeze(K.cast(input_len, tf.int32), axis=1)
#         input_tok = input_tok
#         # input_idx_targ = K.squeeze(K.cast(input_idx_targ, tf.int32), axis=1)
#         # input_idx_targ = K.cast(input_idx_targ, tf.int32)
#         input_mask_LH = K.cast(input_mask_LH, tf.int32)
#         input_mask_RH = K.cast(input_mask_RH, tf.int32)

#         result_raw = self.elmo(
#             inputs={
#                 "tokens": input_tok,
#                 "sequence_len": input_len
#             },
#             signature="tokens",
#             as_dict=True)
#         # result_elmo = result_raw['elmo']
#         # result = None
#         result = result_raw['elmo']

#         # no direction distinction for now
#         # if(self.direction == 'none'):
#         #     result = result_elmo
#         # elif(self.direction == 'forward'):
#         #     result = result_elmo[:, :, :512]                
#         # elif(self.direction == 'backward'):
#         #     result = result_elmo[:, :, 512:]  
#         # else:
#         #     raise NameError("Undefined direction type")
                    
#         mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
#         ret_mask = None
#         if(self.side == "LH"):
#             result = mul_mask(result, tf.cast(input_mask_LH, tf.float32))
#             ret_mask = input_mask_LH
#         elif(self.side == "RH"):
#             result = mul_mask(result, tf.cast(input_mask_RH, tf.float32))
#             ret_mask = input_mask_RH
#         elif(self.side == "tg"):
#             last_idx = tf.equal(input_mask_LH, 1) # True:if the first value of the LH mask is not one
#             last_idx = tf.reduce_sum(tf.cast(last_idx, tf.int32), axis=1) # last idx for non one vector
#             last_idx = tf.stack([tf.range(tf.shape(last_idx)[0]), last_idx], axis=1) # last idx as matrix
#             result = tf.gather_nd(result, last_idx)
#             ret_mask = last_idx

#             # result = tf.gather_nd(result, tf.stack(input_idx_targ, -1))
#             # result = result[:,0,:]
#             # input_batch_mask_tg = tf.not_equal(tf.reduce_sum([input_batch_mask_LH, input_batch_mask_RH], 0), 1)
#             # result = mul_mask(result, tf.cast(input_batch_mask_tg, tf.float32))
#             # tg_idx = 0
#             # result = result[:,input_idx_targ,:]
#         else:
#             result = result

#         return result
    
# #     def compute_mask(self, inputs, mask=None):
# #         return None
# #         if not self.mask_zero:
# #             return None
# #         return K.not_equal(inputs[2], "")
    
#     def compute_output_shape(self, input_shape):
#         if(self.side == 'tg'):
#             return (input_shape[0][0], int(self.dimensions))
#         else:
#             return (input_shape[0][0], input_shape[1][1], int(self.dimensions))
#     #       return (input_shape[0], int(self.dimensions/2))
#     # TODO: clean up; prepare for target word input (word embedding)



# class BertLayer(tf.keras.layers.Layer):
#     def __init__(self, n_fine_tune_layers=10, pooling='first', bert_path=BERT_PATH, **kwargs,):
#         self.n_fine_tune_layers = n_fine_tune_layers
#         self.trainable = False
#         self.output_size = 768
#         self.pooling = pooling
#         self.bert_path = bert_path
#         if self.pooling not in ["first", "mean", "targ", "seq"]:
#             raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling})")
#         super(BertLayer, self).__init__(**kwargs)
        
#     def build(self, input_shape):
#         self.bert = hub.Module(self.bert_path, trainable=self.trainable, name=f"{self.name}_module")
        
#         # remobe unused layers        
#         # select how many layers to fine tune
#         # update trainable vars to contain only the specified layers
#         # add to trainable weights
#         super(BertLayer, self).build(input_shape)
        
#     def call(self, inputs):
#         inputs = [K.cast(x, dtype="int32") for x in inputs]
#         input_ids, input_mask, input_tloc, segment_ids = inputs
#         bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids) # TODO: separate masking between BERT vector fetching vs. model input maksings?
#         mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
                
#         if self.pooling=="first":
#             pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
#         else:
#             result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
#             if self.pooling=="mean":            
#                 # masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
#                 # masked_reduce = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1)    
#                 # mul_mask = lambda x, m: x
#                 input_mask = tf.cast(input_mask, tf.float32)            
#                 # pooled = masked_reduce_mean(result, input_mask)
#                 pooled = mul_mask(result, input_mask)
#             elif self.pooling=="targ":
#                 targ_mask = tf.cast(input_tloc, tf.float32)
#                 pooled = tf.reduce_sum(mul_mask(result, targ_mask), axis=1) / tf.reduce_sum(targ_mask, axis=1, keepdims=True)
#             elif self.pooling=="seq":
#                 pooled = result
#         return pooled
    
#     def compute_output_shape(self, input_shape):
#         return(input_shape[0], self,output_size)