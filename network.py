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
from keras.layers import Dense, Input, Activation, Lambda, concatenate, RepeatVector, multiply, Conv1D, MaxPooling1D, Average
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

    def residual(self, h0, inputs, hidden_size):
        out = Dense(hidden_size, activation='relu')(inputs)
        out = Dropout(0.5)(out)
        res = Lambda(lambda x: x[0]+x[1])([out, h0])
        return res
    
    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size, opt.embd_size, name='word_embd')
            
            word = Input((max_len,), name="word")
            word_embed = embd(word)
            img = Input((2048,), name="image")

            permute_layer = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))
            dot_product = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,2]))
            softmax = Activation('softmax')
            squeeze = Lambda(lambda x: K.squeeze(x, axis=2))

            num_attns = 4
            char_attns = []
            for i in range(num_attns):
                e_char = Dense(1, activation='softmax')(word_embed)
                alpha_char = permute_layer(e_char)
                char_attn = dot_product([word_embed, alpha_char])
                char_attn = squeeze(char_attn)
                char_attns.append(char_attn)

            # concatenate context vector
            char_context = concatenate(char_attns, axis=1)

            h_char = Dense(1)(char_context)
            h_img = Dense(1)(img)
            h_w = concatenate([h_char, h_img], axis=1)
            h_w = softmax(h_w)
            
            # allocate weights
            h_w_charb = Lambda(lambda x: x[:,0], output_shape=(1,))(h_w)
            h_w_img = Lambda(lambda x: x[:,1], output_shape=(1,))(h_w)
            
            h_w_charb = Reshape((1,))(h_w_charb)
            h_w_img = Reshape((1,))(h_w_img)

            h_w_charb = RepeatVector(128*num_attns)(h_w_charb)
            h_w_img = RepeatVector(2048)(h_w_img)

            h_w_char = Reshape((128*num_attns,))(h_w_charb)
            h_w_img = Reshape((2048,))(h_w_img)

            h1_char = multiply([char_context, h_w_char])
            h1_img = multiply([img, h_w_img])

            concat_size = 128*num_attns+2048
            h1 = concatenate([h1_char, h1_img], axis=1)
            h1 = Reshape((concat_size,))(h1)
            h1 = Dropout(0.5)(h1)
           
            res1 = self.residual(h1, h1, concat_size)
            res2 = self.residual(h1, res1, concat_size)
            outputs = Dense(num_classes, activation=activation)(res2)

            model = Model(inputs=[word, img], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))

        return model
