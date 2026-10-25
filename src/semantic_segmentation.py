import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings("ignore")


DATA_DIR = 'data'

metadata_df = pd.read_csv(f'{DATA_DIR}/all_metadata.csv')
train_df = metadata_df[metadata_df['split']=='train']
test_df = metadata_df[metadata_df['split']=='test']

valid_df = train_df.sample(frac=0.1, random_state=42)
train_df = train_df.drop(valid_df.index)

# Augmentations
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
    ], p=0.8),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8)
])

UNLABELED_CLASS = 5  # index of the "Unlabeled" class in the 6-class scheme

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, df, root_dir, batch_size=16, target_size=(256, 256), n_classes=6,
                 augmentation=None, shuffle=False, preprocess=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.shuffle = shuffle
        # preprocess=True feeds the network; preprocess=False returns raw RGB for plotting.
        self.preprocess = preprocess
        self._warned_out_of_range = False
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        # Reshuffle between epochs so batches are not seen in the same order every time.
        if self.shuffle:
            self.df = self.df.sample(frac=1.0).reset_index(drop=True)

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for i, row in batch_df.iterrows():
            img_path = os.path.join(self.root_dir, row['sat_image_path'])
            mask_path = os.path.join(self.root_dir, row['mask_path'])

            # Force 3-channel RGB: some PNGs are RGBA or greyscale.
            img = np.array(Image.open(img_path).convert('RGB'))
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) == 3:
                mask = mask[..., 0]

            # Values outside 0..n_classes-1 would crash one-hot encoding (they come from
            # datasets whose masks use a different label scheme). Treat them as Unlabeled
            # and warn once, so the problem is visible instead of silent.
            out_of_range = mask >= self.n_classes
            if out_of_range.any():
                if not self._warned_out_of_range:
                    print(f"Warning: mask values >= {self.n_classes} found (e.g. {mask_path}); "
                          f"mapping them to Unlabeled. Remap these datasets to the shared classes.")
                    self._warned_out_of_range = True
                mask = mask.copy()
                mask[out_of_range] = UNLABELED_CLASS

            if self.augmentation:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            img = tf.image.resize(img, self.target_size).numpy()
            if self.preprocess:
                # Match the normalisation the ImageNet-pretrained ResNet50 encoder was trained with.
                img = tf.keras.applications.resnet50.preprocess_input(img)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.image.resize(mask, self.target_size, method='nearest')
            mask = tf.squeeze(mask, axis=-1)

            images.append(img)

            # One-hot encode the mask
            mask_one_hot = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)
            masks.append(mask_one_hot)

        return np.array(images), np.array(masks)

def get_dataset(df, root_dir, batch_size=16, target_size=(256, 256), n_classes=6, augmentation=None,
                shuffle=True, preprocess=True):
    dataset = CustomDataset(df, root_dir, batch_size, target_size, n_classes, augmentation,
                            shuffle=shuffle, preprocess=preprocess)
    return dataset

train_dataset = get_dataset(train_df, DATA_DIR, augmentation=augmentation)
valid_dataset = get_dataset(valid_df, DATA_DIR, shuffle=False)
test_dataset = get_dataset(test_df, DATA_DIR, shuffle=False)


def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def total_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return loss + dice

def get_model(name, n_classes=6, input_shape=(256, 256, 3), backbone='resnet50', encoder_weights='imagenet', activation='softmax'):
    resnet50 = tf.keras.applications.ResNet50(weights=encoder_weights, include_top=False, input_shape=input_shape)

    s1 = resnet50.get_layer("conv1_relu").output
    s2 = resnet50.get_layer("conv2_block3_out").output
    s3 = resnet50.get_layer("conv3_block4_out").output
    s4 = resnet50.get_layer("conv4_block6_out").output

    b1 = resnet50.get_layer("conv5_block3_out").output

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    x = layers.UpSampling2D((2, 2))(d4)
    outputs = layers.Conv2D(n_classes, 1, padding="same", activation=activation)(x)

    model = tf.keras.Model(inputs=resnet50.input, outputs=outputs, name="U-Net")
    return model


model = get_model('Unet')

# Learning rate scheduler
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# OneHotMeanIoU takes one-hot labels and softmax outputs and applies argmax internally.
# (Plain MeanIoU expects class indices, so it gave a misleading number on this model's outputs.)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=total_loss,
              metrics=['accuracy', dice_coefficient,
                       tf.keras.metrics.OneHotMeanIoU(num_classes=6, name='mean_iou')])

history = model.fit(
    train_dataset, 
    validation_data=valid_dataset, 
    epochs=100, 
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True),
        lr_schedule
    ]
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])
plt.title('Model IOU Score')
plt.ylabel('IOU Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.show()

# return_dict=True looks metrics up by name. The old code printed results[1], which is
# pixel accuracy, under the label "IOU".
results = model.evaluate(test_dataset, return_dict=True)

print(f"Test mIoU: {results['mean_iou']:.4f}")
print(f"Test Dice: {results['dice_coefficient']:.4f}")
print(f"Test pixel accuracy: {results['accuracy']:.4f}")
print(f"Test loss: {results['loss']:.4f}")

def plot_predictions(model, df):
    n_images = 5

    # Raw RGB for display; the same images preprocessed for the model.
    raw_images, masks = get_dataset(df, DATA_DIR, shuffle=False, preprocess=False)[0]
    model_images = tf.keras.applications.resnet50.preprocess_input(raw_images.copy())
    preds = model.predict(model_images)
    n_images = min(n_images, len(raw_images))
    images = raw_images.astype('uint8')
    
    plt.figure(figsize=(20, 10))
    
    for i in range(n_images):
        plt.subplot(3, n_images, i+1)
        plt.imshow(images[i])
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(3, n_images, i+1+n_images)
        plt.imshow(np.argmax(masks[i], axis=-1), cmap='jet')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(3, n_images, i+1+2*n_images)
        plt.imshow(np.argmax(preds[i], axis=-1), cmap='jet')
        plt.title('Prediction')
        plt.axis('off')
        
    plt.show()

plot_predictions(model, test_df)

