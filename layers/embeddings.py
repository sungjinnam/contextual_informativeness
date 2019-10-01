import tensorflow as tf
import tensorflow_hub as hub
# import keras.backend as K
# from keras.engine import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# ===== ELMo layers using tf.hub =====
# references:
# https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
# https://github.com/AlexYangLi/ABSA_Keras/blob/master/custom_layers.py
# https://github.com/AlexYangLi/ABSA_Keras/blob/4392a52d08640b2b6c6aa586c4468f446fa6d2b7/models.py
ELMO_PATH = 'https://tfhub.dev/google/elmo/2'
BERT_PATH = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"

class ElmoLayer(Layer):
    def __init__(self, side, **kwargs):
        self.dimensions = 1024
        self.trainable = False
        # self.direction = direction
        self.side = side
        # self.mask_zero = mask_zero
        super(ElmoLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):        
        self.elmo = hub.Module(ELMO_PATH, trainable=self.trainable, name="{}_module".format(self.name))
        # if(self.trainable):
        #     self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoLayer, self).build(input_shape)
        
    def call(self, inputs):
        input_len, input_tok, input_mask_LH, input_mask_RH = inputs
        input_len = K.squeeze(K.cast(input_len, tf.int32), axis=1)
        input_tok = input_tok
        # input_idx_targ = K.squeeze(K.cast(input_idx_targ, tf.int32), axis=1)
        # input_idx_targ = K.cast(input_idx_targ, tf.int32)
        input_mask_LH = K.cast(input_mask_LH, tf.int32)
        input_mask_RH = K.cast(input_mask_RH, tf.int32)

        result_raw = self.elmo(
            inputs={
                "tokens": input_tok,
                "sequence_len": input_len
            },
            signature="tokens",
            as_dict=True)
        # result_elmo = result_raw['elmo']
        # result = None
        result = result_raw['elmo']

        # no direction distinction for now
        # if(self.direction == 'none'):
        #     result = result_elmo
        # elif(self.direction == 'forward'):
        #     result = result_elmo[:, :, :512]                
        # elif(self.direction == 'backward'):
        #     result = result_elmo[:, :, 512:]  
        # else:
        #     raise NameError("Undefined direction type")
                    
        mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
        if(self.side == "LH"):
            result = mul_mask(result, tf.cast(input_mask_LH, tf.float32))
        elif(self.side == "RH"):
            result = mul_mask(result, tf.cast(input_mask_RH, tf.float32))
        elif(self.side == "tg"):
            last_idx = tf.not_equal(input_mask_LH, 1) # True:if the first value of the LH mask is not one
            last_idx = tf.reduce_sum(tf.cast(last_idx, tf.int32), axis=1) # last idx for non one vector
            last_idx = tf.stack([tf.range(tf.shape(last_idx)[0]), last_idx], axis=1) # last idx as matrix
            result = tf.gather_nd(result, last_idx)

            # result = tf.gather_nd(result, tf.stack(input_idx_targ, -1))
            # result = result[:,0,:]
            # input_batch_mask_tg = tf.not_equal(tf.reduce_sum([input_batch_mask_LH, input_batch_mask_RH], 0), 1)
            # result = mul_mask(result, tf.cast(input_batch_mask_tg, tf.float32))
            # tg_idx = 0
            # result = result[:,input_idx_targ,:]
        else:
            raise NameError("Undefined side type")

        return result
    
#     def compute_mask(self, inputs, mask=None):
#         return None
#         if not self.mask_zero:
#             return None
#         return K.not_equal(inputs[2], "")
    
    def compute_output_shape(self, input_shape):
        if(self.side == 'tg'):
            return (input_shape[0][0], int(self.dimensions))
        else:
            return (input_shape[0][0], input_shape[1][1], int(self.dimensions))
    #       return (input_shape[0], int(self.dimensions/2))
    # TODO: clean up; prepare for target word input (word embedding)


# ===== BERT layers using tf.hub =====
# reference: https://github.com/strongio/keras-bert/blob/master/keras-bert.ipynb
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, pooling='first', bert_path=BERT_PATH, **kwargs,):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = False
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling})")
        super(BertLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path, trainable=self.trainable, name=f"{self.name}_module")
        
        # remobe unused layers        
        # select how many layers to fine tune
        # update trainable vars to contain only the specified layers
        # add to trainable weights
        super(BertLayer, self).build(input_shape)
        
    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids) # TODO: separate masking between BERT vector fetching vs. model input maksings?
        
        if self.pooling=="first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
        elif self.pooling=="mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            # masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            # masked_reduce = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1)    
            # mul_mask = lambda x, m: x
            input_mask = tf.cast(input_mask, tf.float32)            
            # pooled = masked_reduce_mean(result, input_mask)
            pooled = mul_mask(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling})")
        
        return pooled
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0], self,output_size)
        