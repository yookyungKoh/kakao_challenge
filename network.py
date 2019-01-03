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
from keras.layers import Dense, Input, Activation, Lambda, concatenate, Average, RepeatVector, multiply, Conv1D, MaxPooling1D
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout

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
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')

            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

            embd_out = Dropout(rate=0.5)(uni_embd)
            relu = Activation('relu', name='relu1')(embd_out)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class ImgTextFusion():
    def __init__(self, output_dim, num_anchor, img_layer_name='img_fusion_layer', txt_layer_name='txt_fusion_layer'):
        self.img_layer = Dense(output_dim, name=img_layer_name)
        self.txt_layer = Dense(output_dim, name=txt_layer_name)
        self.anchor = Dense(num_anchor, use_bias=False, activation='softmax')

    def __call__(self, x):
        img, txt = x
        img = self.img_layer(img)
        txt = self.txt_layer(txt)
        img_txt_fusion = Average()([img, txt])
        y_img = self.anchor(img)
        y_txt = self.anchor(txt)
        return img_txt_fusion, y_img, y_txt
