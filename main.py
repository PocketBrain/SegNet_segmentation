import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import GlobalAveragePooling2D, UpSampling2D, Conv2D, MaxPooling2D
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.losses import binary_crossentropy


path= "D:\\datasets\\"
train = os.listdir(path + "train_v2\\")
test = os.listdir(path + "test_v2\\")
df = pd.read_csv(path + "train_ship_segmentations_v2.csv")
ImgId = '90c4c298c.jpg'
IMG_SCALING = (1, 1)

def decoding(mask_rle, shape=(768, 768)):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lenghts = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lenghts
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    img = img.reshape(shape).T
    return img

def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += decoding(mask)
    return np.expand_dims(all_masks, -1)


img = cv2.imread(path + "train_v2\\" + ImgId)
img_masks = df.loc[df['ImageId'] == ImgId, 'EncodedPixels'].tolist()


allMask = np.zeros((768, 768))
for mask in img_masks:
    allMask += decoding(mask)

fig, array = plt.subplots(1, 3, figsize=(15, 40))
array[0].axis('off')
array[1].axis('off')
array[2].axis('off')
array[0].imshow(img[..., [2, 1, 0]])  #rgb
array[1].imshow(allMask, cmap='magma')
array[2].imshow(img, cmap='magma')
array[2].imshow(allMask, alpha=0.4, cmap='magma')
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

masks = df.drop(df[df['EncodedPixels'].isnull()].sample(70000,random_state=42).index)

unique_img_ids = masks.groupby('ImageId').size().reset_index(name='Counts')

train_ids, valid_ids = train_test_split(unique_img_ids,
                 test_size = 0.05,
                 stratify = unique_img_ids['Counts'],
                 random_state=42)

train_df = pd.merge(df, train_ids)
valid_df = pd.merge(df, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


train_df['Counts'] = train_df.apply(lambda c_row: c_row['Counts'] if
                                    isinstance(c_row['EncodedPixels'], str) else
                                    0, 1)
valid_df['Counts'] = valid_df.apply(lambda c_row: c_row['Counts'] if
                                    isinstance(c_row['EncodedPixels'], str) else
                                    0, 1)

def mask2image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = float)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += decoding(mask)
    return np.expand_dims(all_masks, -1)



def keras_generator(gen_df, batch_size=4):
    all_batches = list(gen_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = cv2.imread(path + 'train_v2\\'+ c_img_id)
            c_img = rgb_path
            c_mask = mask2image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

train_gen = keras_generator(train_df,5)
train_x, train_y = next(train_gen)
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())


inp = Input(shape=(768,768,3))

conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
conv_1_1 = Activation('relu')(conv_1_1)

conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
conv_1_2 = Activation('relu')(conv_1_2)

pool_1 = MaxPooling2D(2)(conv_1_2)

conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
conv_2_1 = Activation('relu')(conv_2_1)

conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
conv_2_2 = Activation('relu')(conv_2_2)

pool_2 = MaxPooling2D(2)(conv_2_2)


conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
conv_3_1 = Activation('relu')(conv_3_1)

conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
conv_3_2 = Activation('relu')(conv_3_2)

pool_3 = MaxPooling2D(2)(conv_3_2)


conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
conv_4_1 = Activation('relu')(conv_4_1)

conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
conv_4_2 = Activation('relu')(conv_4_2)

pool_4 = MaxPooling2D(2)(conv_4_2)


conv_5_1 = Conv2D(512, (3, 3), padding='same')(pool_4)
conv_5_1 = Activation('relu')(conv_5_1)

conv_5_2 = Conv2D(512, (3, 3), padding='same')(conv_5_1)
conv_5_2 = Activation('relu')(conv_5_2)

pool_5 = MaxPooling2D(2)(conv_4_2)


up_1 = UpSampling2D(2, interpolation='bilinear')(pool_5)
conv_up_1_1 = Conv2D(512, (3, 3), padding='same')(up_1)
conv_up_1_1 = Activation('relu')(conv_up_1_1)

conv_up_1_2 = Conv2D(512, (3, 3), padding='same')(conv_up_1_1)
conv_up_1_2 = Activation('relu')(conv_up_1_2)


up_2 = UpSampling2D(2, interpolation='bilinear')(pool_4)
conv_up_2_1 = Conv2D(256, (3, 3), padding='same')(up_2)
conv_up_2_1 = Activation('relu')(conv_up_2_1)

conv_up_2_2 = Conv2D(256, (3, 3), padding='same')(conv_up_2_1)
conv_up_2_2 = Activation('relu')(conv_up_2_2)


up_3 = UpSampling2D(2, interpolation='bilinear')(pool_3)
conv_up_3_1 = Conv2D(128, (3, 3), padding='same')(up_3)
conv_up_3_1 = Activation('relu')(conv_up_3_1)

conv_up_3_2 = Conv2D(128, (3, 3), padding='same')(conv_up_3_1)
conv_up_3_2 = Activation('relu')(conv_up_3_2)


up_4 = UpSampling2D(2, interpolation='bilinear')(pool_2)
conv_up_4_1 = Conv2D(64, (3, 3), padding='same')(up_4)
conv_up_4_1 = Activation('relu')(conv_up_4_1)

conv_up_4_2 = Conv2D(64, (3, 3), padding='same')(conv_up_4_1)
conv_up_4_2 = Activation('relu')(conv_up_4_2)


up_5 = UpSampling2D(2, interpolation='bilinear')(conv_up_4_2)
conv_up_5_1 = Conv2D(32, (3, 3), padding='same')(up_5)
conv_up_5_1 = Activation('relu')(conv_up_5_1)

conv_up_5_2 = Conv2D(1, (3, 3), padding='same')(conv_up_5_1)
result = Activation('sigmoid')(conv_up_5_2)


segnet_model = Model(inputs=inp, outputs=result)
segnet_model.summary()


first = keras.callbacks.ModelCheckpoint('first_segnet.w',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_freq=1)

last = keras.callbacks.ModelCheckpoint('last_segnet.w',
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=False,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq=1)

report = [first, last]

def num(y_true, y_pred, eps=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection + eps) / (union + eps), axis=0)


def zero(y_true, y_pred):
    return num(1 - y_true, 1 - y_pred)


def agg(in_gt, in_pred):
    return -1e-2 * zero(in_gt, in_pred) - num(in_gt, in_pred)


adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

segnet_model.compile(optimizer=adam, loss=agg, metrics=[num, zero, 'binary_accuracy'])

loss_history = segnet_model.fit(keras_generator(train_df),
                                steps_per_epoch=100,
                                epochs=5,
                                validation_data=keras_generator(valid_df),
                                validation_steps=50)

