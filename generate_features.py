'''
在意图估计模型的编码器起始部分，使用了VGG16预训练网络。
所以，原始图像需经过处理并送入VGG16网络得到特征图（7,7,255）。
该脚本实现了上述操作，并分为裁剪区域是否2倍于矩形框两种情况，用参数进行选择
特征图命名规范为“帧号_行人id.pkl”，
保存路径为/dataset/data/pie/intention/...
'''
import os,sys
from pie_data import PIE
from pie_intent_without_bbox import *
from keras.applications import vgg16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

def get_path(self,
             type_save='models',  # model or data
             models_save_folder='',
             model_name='convlstm_encdec',
             file_name='',
             data_subset='',
             data_type='',
             save_root_folder=os.environ['PIE_PATH'] + '/data/'):
    """
    A path generator method for saving model and config data. Creates directories
    as needed.
    :param type_save: Specifies whether data or model is saved.
    :param models_save_folder: model name (e.g. train function uses timestring "%d%b%Y-%Hh%Mm%Ss")
    :param model_name: model name (either trained convlstm_encdec model or vgg16)
    :param file_name: Actual file of the file (e.g. model.h5, history.h5, config.pkl)
    :param data_subset: train, test or val
    :param data_type: type of the data (e.g. features_context_pad_resize)
    :param save_root_folder: The root folder for saved data.
    :return: The full path for the save folder
    """
    assert (type_save in ['models', 'data'])
    if data_type != '':
        assert (any([d in data_type for d in ['images', 'features']]))
    root = os.path.join(save_root_folder, type_save)

    if type_save == 'models':
        save_path = os.path.join(save_root_folder, 'pie', 'intention', models_save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path
    else:
        save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

def load_images_and_process(data,convnet,data_subset='train',**data_opts):
    save_path=get_path(
        type_save='data',
        data_type='features' + '_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
        model_name='vgg16_' + 'none',
        data_subset=data_subset
    )

    print("Generating {} features crop_type={} crop_mode=pad_resize \nsave_path={}, "\
          .format(data_subset,data_opts['crop_type'], save_path))

    img_sequences=data['image']
    bbox_sequences=data['bbox']
    ped_ids=data['ped_id']

    i = -1
    for seq, pid in zip(img_sequences, ped_ids):
        i += 1
        update_progress(i / len(img_sequences))
        for imp, b, p in zip(seq, bbox_sequences[i], pid):
            set_id = imp.split('/')[-3]
            vid_id = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]
            img_save_folder = os.path.join(save_path, set_id, vid_id)
            img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')
            img_data = load_img(imp)  # 加载PIL图像实例,一张
            bbox = jitter_bbox(imp, [b], 'enlarge', 2)[0]  # 扩大裁减范围
            bbox = squarify(bbox, 1, img_data.size[0])  # 调整宽高比
            bbox = list(map(int, bbox[0:4]))  # 把浮点数取整
            cropped_image = img_data.crop(bbox)  # 裁剪图像
            img_data = img_pad(cropped_image, mode='pad_resize', size=224)
            image_array = img_to_array(img_data)
            preprocessed_img = vgg16.preprocess_input(image_array)
            expanded_img = np.expand_dims(preprocessed_img, axis=0)
            img_features = convnet.predict(expanded_img)
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder)
            with open(img_save_path, 'wb') as fid:
                pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

def _generate_features(crop_type='same'):
    data_opts={
        'crop_type':crop_type,
        'crop_mode': 'pad_resize',
        'data_type':['bbox'],
        'fstride':1,
        'seq_type':'intention',
        'max_size_observe':1,
        'min_track_size':1, #小于该帧数的行人将丢弃
        'seq_overlap_rate':0.5,
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': 'default',  # kfold, random, default
        'sample_type': 'all',
    }

    t=PIEIntent()
    imdb=PIE(data_path=os.environ.copy()['PIE_PATH'])
    convnet=vgg16.VGG16(input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet')

    beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
    beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

    load_images_and_process(beh_seq_val,convnet,'val',**data_opts)
    load_images_and_process(beh_seq_train,convnet,'train', **data_opts)
    load_images_and_process(beh_seq_test,convnet,'test', **data_opts)


if __name__=='__main__':
    try:
        crop_type=sys.argv[1]
        if crop_type=='all':
            _generate_features(crop_type='same')
            _generate_features(crop_type='context')
        else:
            _generate_features(crop_type=crop_type)
    except ValueError:
        raise ValueError(
            'Usage: python generate_features.py <crop_type>\n'
            'crop_type: \'same\', \'context\' \'all\' \n'
        )
