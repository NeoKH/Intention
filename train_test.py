"""
The code implementation of the paper:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import os
import sys

from pie_intent import PIEIntent
# from pie_predict import PIEPredict

from pie_data import PIE

import keras.backend as K
import tensorflow as tf

from prettytable import PrettyTable

dim_ordering = K.image_dim_ordering()

#train models with data up to critical point
#only for PIE
#train_test = 0 (train only), 1 (train-test), 2 (test only)
def train_intent(train_test=1):

    data_opts = {
            'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  #  kfold, random, default
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': 0, #  discard tracks that are shorter
            'max_size_observe': 15,  # number of observation frames
            'max_size_predict': 5,  # number of prediction frames
            'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
            'balance': True,  # balance the training and testing samples
            'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
            'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
            'encoder_input_type': [],
            'decoder_input_type': ['bbox'],
            'output_type': ['intention_binary']
            }


    t = PIEIntent(num_hidden_units=128,
                  regularizer_val=0.001,
                  lstm_dropout=0.4,
                  lstm_recurrent_dropout=0.2,
                  convlstm_num_filters=64,
                  convlstm_kernel_size=2)

    saved_files_path = ''

    imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])

    #pretrained_model_path = 'data/pie/intention/context_loc_pretrained'
   # pretrained_model_path='data/models/pie/intention/imgcontext/14Mar2020-05h58m05s'

   # input_types=['imgbbox','imgcontext','imgbbox+loc','imgcontext+loc']
    input_types=['imgbbox']
    for input_type in input_types:
        print('input_type is ',input_type)
        
        if input_type=='imgbbox':
            data_opts['crop_type']='same'
            data_opts['decoder_input_type']=[]
            
        elif input_type=='imgcontext':
            data_opts['crop_type']='context'
            data_opts['decoder_input_type']=[]
        elif input_type=='imgbbox+loc':
            data_opts['crop_type']='same'
            data_opts['decoder_input_type']=['bbox']
            #pretrained_model_path='data/models/pie/intention/imgbbox+loc/12Mar2020-11h30m34s'
        else:
            data_opts['crop_type']='context'
            data_opts['decoder_input_type']=['bbox']
            #pretrained_model_path='data/models/pie/intention/imgcontext+loc/12Mar2020-14h11m15s'
        
        if train_test < 2:  # Train
            beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
            #beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary',ratio=1)

            beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
            #beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary',ratio=1)

            saved_files_path = t.train(data_train=beh_seq_train,
                                    data_val=beh_seq_val,
                                    epochs=400,
                                    loss=['binary_crossentropy'],
                                    metrics=['accuracy'],
                                    batch_size=1,
                                    optimizer_type='rmsprop',
                                    data_opts=data_opts,
                                    input_type=input_type)

            print(data_opts['seq_overlap_rate'])

        if train_test > 0:  # Test
            if saved_files_path == '':
                saved_files_path = pretrained_model_path

            beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
            acc, f1 = t.test_chunk(beh_seq_test, data_opts, saved_files_path, False)
            
            pt = PrettyTable(['Acc', 'F1'])
            pt.title = 'Intention model ('+input_type+')'
            pt.add_row([acc, f1])
            
            print(pt)

            K.clear_session()
            tf.reset_default_graph()
           # return saved_files_path
 
if __name__ == '__main__':
    try:
        train_test = int(sys.argv[1])
        train_intent(train_test=train_test)
    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test> <input_type>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n'
                        )
