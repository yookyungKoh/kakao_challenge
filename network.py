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
        max_len = opt.max_len
        max_len_b = opt.max_len_b # 5
        max_len_m = opt.max_len_m # 5
        max_len_c = opt.max_len_c # 200
        voca_size = opt.unigram_hash_size + 1

        kernel_size = [1, 3, 5, 7]
        num_kernels = [50, 50, 50, 50]
        kernel_params = list(zip(kernel_size, num_kernels))

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size, opt.embd_size, name='word_embd')
            
            word = Input((max_len,), name="word")
            word_embed = embd(word)  # (L, 128)
            char = Input((max_len_c,), name="char")
            char_embed = embd(char) #(200, 128)
            brand = Input((max_len_b,), name="brand")
            brand_embed = embd(brand) #(5, 128)
#            maker = Input((max_len_m,), name="maker")
#            price = Input((1,), name="price")
            img = Input((2048,), name="image")

            permute_layer = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))
            dot_product = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,2]))
            softmax = Activation('softmax')
            residual = Lambda(lambda x: x[0]+x[1])
            squeeze = Lambda(lambda x: K.squeeze(x, axis=2))
            
            # CNN
            conv_outputs = []
            for kernel_size, num_kernel in kernel_params:
                conv_output = Conv1D(num_kernel, kernel_size, padding='valid', activation='relu')(char_embed) #(N, L, 50)
                conv_output = MaxPooling1D(pool_size=4)(conv_output)
                conv_output = Lambda(lambda x: x[:,:48,:], output_shape=(48,50))(conv_output)
                conv_outputs.append(conv_output)
            char_out = concatenate(conv_outputs, axis=2) #(N, 48, 200)
            
            e_char = Dense(1, activation='tanh')(char_out) #(N, 48, 1)
            alpha_char = softmax(e_char)
            alpha_char = permute_layer(alpha_char)
            char_attn = dot_product([char_out, alpha_char]) #(N, 200)

            e_word = Dense(1, activation='tanh')(word_embed) #(N, 50, 1)
            alpha_word = softmax(e_word)
            alpha_word = permute_layer(alpha_word)
            word_attn = dot_product([word_embed, alpha_word]) #(N, 128)
            
            e_brand = Dense(1, activation='tanh')(brand_embed) #(N, 5, 1)
            alpha_brand = softmax(e_brand)
            alpha_brand = permute_layer(alpha_brand)
            brand_attn = dot_product([brand_embed, alpha_brand]) #(N, 128)
    
            h_word = Dense(1)(squeeze(word_attn))
            h_char = Dense(1)(squeeze(char_attn))
            h_brand = Dense(1)(squeeze(brand_attn))
            h_img = Dense(1)(img)

            h_w = concatenate([h_word, h_char, h_brand, h_img], axis=1) #(N, 4)
            h_w = softmax(h_w)
            
            # allocate weights
            h_w_word = Lambda(lambda x: x[:,0], output_shape=(1,))(h_w)
            h_w_char = Lambda(lambda x: x[:,1], output_shape=(1,))(h_w)
            h_w_brand = Lambda(lambda x: x[:,2], output_shape=(1,))(h_w)
            h_w_img = Lambda(lambda x: x[:,3], output_shape=(1,))(h_w)
            
            h_w_word = Reshape((1,))(h_w_word)
            h_w_char = Reshape((1,))(h_w_char)
            h_w_brand = Reshape((1,))(h_w_brand)
            h_w_img = Reshape((1,))(h_w_img)

            h_w_word = RepeatVector(128)(h_w_word)
            h_w_char = RepeatVector(200)(h_w_char)
            h_w_brand = RepeatVector(128)(h_w_brand)
            h_w_img = RepeatVector(2048)(h_w_img)

            h_w_word = Reshape((128,))(h_w_word)
            h_w_char = Reshape((200,))(h_w_char)
            h_w_brand = Reshape((128,))(h_w_brand)
            h_w_img = Reshape((2048,))(h_w_img)

            h1_word = multiply([h_word, h_w_word])
            h1_char = multiply([h_char, h_w_char])
            h1_brand = multiply([h_brand, h_w_brand])
            h1_img = multiply([img, h_w_img])

            concat_size = 128+200+128+2048
            h1 = concatenate([h1_word, h1_char, h1_brand, h1_img], axis=1) #(N, 2504)
            h1 = Reshape((concat_size,))(h1)
            h1 = Dropout(0.5)(h1)
            h2 = Dense(concat_size, activation='relu')(h1)
            h2 = Dropout(0.5)(h2)
            res1 = residual([h2, h1])
            h3 = Dense(concat_size, activation='relu')(res1)
            h3 = Dropout(0.5)(h3)
            res2 = residual([h3, h1])
            outputs = Dense(num_classes, activation=activation)(res2)

            model = Model(inputs=[word, char, brand, img], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)
            model.compile(loss='binary_crossentropy',
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

        word_input = Input((word_max_len,), name="input_word")
        word_count_input = Input((word_max_len,), name="input_word_count")
        img_input = Input((img_size,), name="input_img")
        char_input = Input((char_max_len,), name='imput_char')
        

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