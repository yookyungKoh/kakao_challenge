# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import *
from keras.layers.embeddings import Embedding

from keras.initializers import Ones, Zeros
from keras.engine.topology import Layer

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len # 50
        max_len_c = opt.max_len_c # 300
        img_size = 2048
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size, opt.embd_size, name='word_embd', mask_zero=True)
            
            ngram_input = Input((max_len_c,), name="input_ngram")
            word_input = Input((max_len,), name="input_word")
            word_count_input = Input((max_len,), name="input_word_count")
            img_input = Input((img_size,), name="input_img")

            # image
            img_dropout = Dropout(rate=0.5)(img_input)
            # image end

            # word
            word_embd = embd(word_input)
            word_count_mat = Reshape((max_len, 1))(word_count_input)  # weight
            word_mat = dot([word_embd, word_count_mat], axes=1)
            word = Reshape((opt.embd_size, ))(word_mat)
            word_out = Dropout(rate=0.5)(word)
            word_relu = Activation('relu', name='word_relu')(word_out)
            # word end

            # ngram
            num_q = 4
            ngram_embd = embd(ngram_input)  # (batch_size, max_len_c, 128)
            e_ngram = Dense(100*num_q, use_bias=False)(ngram_embd)  # (batch_size, max_len_c, 100*num_q)
            def reshape_e_ngram(x):
                s = tf.shape(x)   # (batch_size, max_len_c, 100*num_q)
                x = tf.reshape(x, [s[0], s[1], 100, num_q])  #  (batch_size, max_len_c, 100, num_q)
                x = tf.transpose(x, [3, 0, 1, 2])
                x = tf.reshape(x, [-1, s[1], 100])  # (num_q * batch_size, max_len_c, 100)
                return x
            e_ngram = Lambda(reshape_e_ngram, name="reshape_e_ngram")(e_ngram)  # (num_q * batch_size, max_len_c, 100)
            # e_ngram = Lambda(lambda x: x / np.sqrt(x), name="scale_dot")(e_ngram)
            attn_ngram = Softmax(axis=1)(e_ngram)  # (num_q * batch_size, max_len_c, 100)
            avg_attn_ngram = Lambda(lambda x: K.mean(x, axis=2), name="e_ngram_mean")(attn_ngram)  # (num_q * batch_size, max_len_c)
            def reshape_e_ngram2(x):
                s = tf.shape(x)   # (num_q * batch_size, max_len_c)
                x = tf.reshape(x, [num_q, -1, s[1]])  #  (num_q, batch_size, max_len_c)
                x = tf.transpose(x, [1, 2, 0])
                x = tf.reshape(x, [-1, s[1], num_q])  # (batch_size, max_len_c, num_q)
                return x
            avg_attn_ngram = Lambda(reshape_e_ngram2, name="reshape_e_ngram2")(avg_attn_ngram)  # (batch_size, max_len_c, num_q)
            context_ngram = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]), name='ngram_context')([ngram_embd, avg_attn_ngram])  # (batch_size, 128, num_q)
            context_ngram = Flatten(name="context_flatten")(context_ngram)
            context_ngram = Dropout(rate=0.5, name="context_ngram_dropout")(context_ngram)
            context_ngram = Activation('relu', name="context_ngram_relu")(context_ngram)
            # ngram_end

            concat = Concatenate()([context_ngram, img_dropout, word_relu])
            pred = Dense(num_classes, activation='softmax', name="pred")(concat)

            model = Model(inputs=[ngram_input, word_input, word_count_input, img_input], outputs=pred)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optm,
                          metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

class TextImgFusion():
    def __init__(self):
        self.logger = get_logger('text_img_fusion')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        img_size = 2048

        word_input = Input((max_len,), name="input_word")
        word_count_input = Input((max_len,), name="input_word_count")
        img_input = Input((img_size,), name="input_img")

class CharCNN():
    def __init__(self):
        self.logger = get_logger('charcnn')

    def get_model(self, num_classes):
        word_max_len = opt.max_len
        char_max_len = opt.max_len_c
        voca_size = opt.unigram_hash_size + 1
        img_size = 2048

        with tf.device('/gpu:0'):
            word_input = Input((word_max_len,), name="input_word")
            word_count_input = Input((word_max_len,), name="input_word_count")
            char_input = Input((char_max_len,), name='imput_char')
            img_input = Input((img_size,), name="input_img")

            embd = Embedding(voca_size, opt.embd_size, name='word_embd', mask_zero=True)
            word_embd = embd(word_input)  # (batch_size, word_max_len, 128)
            char_embd = embd(char_input)  # (batch_size, char_max_len, 128)

            # Conv1D()

class TextSelfAttentionImg:
    def __init__(self):
        self.logger = get_logger('char self_attention + img')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd',
                             mask_zero=True)
            text_input = Input((max_len,), name="input_text")
            img_input = Input((2048,), name="input_image")
            text_embd = embd(text_input)

            n_head = 8
            model_dim = 128
            n_decoder_querys = 4
            subspace_dim = model_dim // n_head
            mha_outputs, attn = MultiHeadAttention(n_head, model_dim, d_k=subspace_dim, d_v=subspace_dim, dropout=0.1)(text_embd, text_embd, text_embd)
            mha_outputs = PositionwiseFeedForward(128, 512)(mha_outputs)  # [batch_size, len_k, d_model]
            decoder_attention = TimeDistributed(Dense(n_decoder_querys, use_bias=False))(mha_outputs)
            decoder_attention = Lambda(lambda x: K.softmax(x, axis=1))(decoder_attention)
            decoder_context = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,1]))([mha_outputs, decoder_attention])
            decoder_context = Flatten()(decoder_context)
            x = Concatenate()([decoder_context, img_input])
            x = Dense(num_classes, activation=activation)(x)
            model = Model(inputs=[text_input, img_input], outputs=x)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                            optimizer=optm,
                            metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
									 initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
									initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

class ScaledDotProductAttention():
	def __init__(self, d_k, attn_dropout=0.1):
		self.temper = np.sqrt(d_k)
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
	def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
		self.mode = mode
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		self.dropout = dropout
		if mode == 0:
			self.qs_layer = Dense(n_head*d_k, use_bias=False)
			self.ks_layer = Dense(n_head*d_k, use_bias=False)
			self.vs_layer = Dense(n_head*d_v, use_bias=False)
		elif mode == 1:
			self.qs_layers = []
			self.ks_layers = []
			self.vs_layers = []
			for _ in range(n_head):
				self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
		self.attention = ScaledDotProductAttention(d_model)
		self.layer_norm = LayerNormalization() if use_norm else None
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, mask=None):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		if self.mode == 0:
			qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
			ks = self.ks_layer(k)
			vs = self.vs_layer(v)

			def reshape1(x):
				s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
				x = tf.reshape(x, [s[0], s[1], n_head, d_k])
				x = tf.transpose(x, [2, 0, 1, 3])  
				x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
				return x
			qs = Lambda(reshape1)(qs)
			ks = Lambda(reshape1)(ks)
			vs = Lambda(reshape1)(vs)

			if mask is not None:
				mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
			head, attn = self.attention(qs, ks, vs, mask=mask)  
				
			def reshape2(x):
				s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
				x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
				x = tf.transpose(x, [1, 2, 0, 3])
				x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
				return x
			head = Lambda(reshape2)(head)
		elif self.mode == 1:
			heads = []; attns = []
			for i in range(n_head):
				qs = self.qs_layers[i](q)   
				ks = self.ks_layers[i](k) 
				vs = self.vs_layers[i](v) 
				head, attn = self.attention(qs, ks, vs, mask)
				heads.append(head); attns.append(attn)
			head = Concatenate()(heads) if n_head > 1 else heads[0]
			attn = Concatenate()(attns) if n_head > 1 else attns[0]

		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		if not self.layer_norm: return outputs, attn
		outputs = Add()([outputs, q])
		return self.layer_norm(outputs), attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)