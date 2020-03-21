import os
import sys
from pie_intent import PIEIntent
from pie_data import PIE
import keras.backend as K
import tensorflow as tf
from prettytable import PrettyTable

dim_ordering = K.image_dim_ordering()

if __name__ == '__main__':
    ratio = float(sys.argv[1])
    data_opts = {
        'fstride': 1,
        'sample_type': 'all',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': 'default',  # kfold, random, default
        'seq_type': 'intention',  # crossing , intention
        'min_track_size': 0,  # discard tracks that are shorter
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
    input_types = ['imgbbox']
    for input_type in input_types:
        print('input_type is ', input_type)

        if input_type == 'imgbbox':
            data_opts['crop_type'] = 'same'
            data_opts['decoder_input_type'] = []

        elif input_type == 'imgcontext':
            data_opts['crop_type'] = 'context'
            data_opts['decoder_input_type'] = []
        elif input_type == 'imgbbox+loc':
            data_opts['crop_type'] = 'same'
            data_opts['decoder_input_type'] = ['bbox']
            # pretrained_model_path='data/models/pie/intention/imgbbox+loc/12Mar2020-11h30m34s'
        else:
            data_opts['crop_type'] = 'context'
            data_opts['decoder_input_type'] = ['bbox']
            # pretrained_model_path='data/models/pie/intention/imgcontext+loc/12Mar2020-14h11m15s'

        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary',ratio=ratio)
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary',ratio=ratio)
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
        K.clear_session()
        tf.reset_default_graph()

