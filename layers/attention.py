# https://github.com/thushv89/attention_keras/blob/master/examples/nmt/model.py
import tensorflow as tf
import os
# from tensorflow.python.keras.layers import Layer
# from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
# from keras.engine import Layer
# import keras.backend as K
from tensorflow.keras import layers

class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]
    
    
class AttentionMulMask(Layer):
    # adding attention masks to tf.keras' attention layer
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(AttentionMulMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
        super(AttentionMulMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        in_vec_cntx, in_vec_targ, mask_cntx, mask_targ, max_seq_len = inputs
            # global attention of context words in the sentence w.r.t. the target word
        def attention_block_mul(in_vec_cntx, in_vec_targ, mask_cntx, mask_targ, max_seq_len):
            tg_emb_repeat = layers.RepeatVector(max_seq_len, name='targ')(in_vec_targ)
        #     att_fout = layers.AdditiveAttention(name="attention_fout")(inputs=[in_vec_cntx, tg_emb_repeat], 
        #                                                        mask=[tf.cast(mask_cntx, tf.bool), tf.cast(mask_targ, tf.bool)])    
            att_fout = layers.Attention(name="attention_fout")(inputs=[in_vec_cntx, tg_emb_repeat], 
                                                               mask=[tf.cast(mask_cntx, tf.bool), tf.cast(mask_targ, tf.bool)])
            att_sfmx = layers.Dense(max_seq_len, use_bias=False, activation='softmax', name="attention_sfmx")(att_fout)

            # selecting the softmax result slice; the output shape should be::: shape=(?, 30, 1)
            non_targ_idx = tf.where(tf.not_equal(mask_cntx, 0))[0][0]
            att_sfmx = tf.expand_dims(att_sfmx[:, non_targ_idx, :], -1) 

            # masking the softmax output - attention weights for the context words are included only
            att_sfmx = layers.Lambda(lambda x:tf.where(tf.cast(tf.expand_dims(x[0], -1), tf.bool), x[1], tf.zeros_like(x[1])), name="attention_sfmx_out")([mask_cntx, att_sfmx])

            # normalizing the attention weights
            att_sfmx = layers.Lambda(lambda x:x/tf.expand_dims(tf.reduce_sum(x, axis=1), -1), name="attention_sfmx_nrm")(att_sfmx)

            # weighted sentence embedding vector 
            att_out = layers.multiply([att_fout, att_sfmx], name="attention_wvec")
            # att_out = layers.GlobalAveragePooling1D(name="attention_out")(att_out)
            return(att_sfmx, att_out)
        
        return(attention_block_mul(in_vec_cntx, in_vec_targ, mask_cntx, mask_targ, max_seq_len))

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

    
    
        
    
