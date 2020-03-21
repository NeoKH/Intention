import os
import sys
from pie_intent import PIEIntent
from pie_data import PIE
import keras.backend as K
import tensorflow as tf
from prettytable import PrettyTable

dim_ordering = K.image_dim_ordering()


def count_ratio(data):
    pos = 0
    output = data['output']
    total = output.shape[0]
    for output_seq in output:
        pos += output_seq[0][0]
    return pos, total - pos

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

        data_train=t.get_train_val_data(beh_seq_train, data_type=['intention_binary'], seq_length=15, overlap=0.5)
        data_val=t.get_train_val_data(beh_seq_val, data_type=['intention_binary'], seq_length=15, overlap=0.5)

        pos_train,neg_train=count_ratio(data_train)
        pos_val, neg_val = count_ratio(data_val)

        pt = PrettyTable(['data_type','positive samples', 'negative samples','ratio'])
        pt.title = 'Proportion of positive and negative samples'
        pt.add_row(['train',pos_train, neg_train, pos_train/neg_train])
        pt.add_row(['val',pos_val, neg_val, pos_val/neg_val])
        pt.add_row(['seq_length=15, overlap=0.5'])

        K.clear_session()
        tf.reset_default_graph()

