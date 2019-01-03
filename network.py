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
from keras.layers import Dense, Input, Activation, Lambda, concatenate, RepeatVector, multiply, Conv1D, MaxPooling1D
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout

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
