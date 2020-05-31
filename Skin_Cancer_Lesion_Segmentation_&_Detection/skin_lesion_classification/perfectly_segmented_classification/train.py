from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras import optimizers
import numpy as np
import csv
from scipy.misc import imresize
import os
import h5py

# Options
TRAIN = 1

### paths to save results
model_name = "segmented_skin_classifier"
model_path = "models_trained/" +model_name+"/"

### paths to training and testing data
train_data_dir = '/imatge/aromero/work/image-classification/isbi-perfectly-segmented-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-perfectly-segmented-dataset/val'

### paths to weight files
weights_path = '/imatge/aromero/work/image-classification/weights/vgg16_weights.h5'  # this is the pretrained vgg16 weights
top_model_weights_path = '/imatge/aromero/work/image-classification/MIDDLE_Perfectly_Segmented_Lesion_Classification/weights/bottleneck_model_weights.h5'   # this is the best performing model before fine tuning

f_hist_1 = open(model_path+model_name+"_hist_bottleneck.txt", 'w')
f_hist_2 = open(model_path+model_name+"_hist_finetuning.txt", 'w')

### other hyperparameters
nb_train_samples = 900				# Training samples
nb_train_samples_benign = 727		# Testing samples
nb_train_samples_malignant = 173	# Malignant Training samples
nb_validation_samples = 378			# Malignant Training samples
nb_validation_samples_benign = 303	# Benign Training samples
nb_validation_samples_maligant = 75	# Malignant Testing samples
nb_epoch = 50
img_width, img_height = 224, 224
class_weights={0:1.,
               1:4.2 }

# (you'll have to divide up the dataset into the right directories to match this setup
# since the kaggle dataset doesn't come with a validation split)

early_stopping = EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='auto')
# ^^ this stops training after validation loss stops improving

# checkpoint
checkpoint_path="/imatge/aromero/work/image-classification/MIDDLE_Segmented_Lesion_Classification/weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
save_best_only = [checkpoint]

# Load Data
def load_data(data_type):
    '''Load and resize data'''
    print 'Loading data: ', data_type
    if data_type == 'train':
        data_dir = train_data_dir
        print 'Loading train data... '
    else:
        data_dir = validation_data_dir
        print 'Loading test data... '

    # assume malignant = 0, benign = 1
    malignant_path = os.path.join(data_dir, 'malignant')
    malignant_list = os.listdir(malignant_path)  # get a list of all malignant image files in directory
    malignant_num = len(malignant_list)
    benign_path = os.path.join(data_dir, 'benign')
    benign_list = os.listdir(benign_path)
    benign_num = len(benign_list)

    _X = np.empty((benign_num + malignant_num, 3, img_width, img_height), dtype='float32')
    _y = np.zeros((benign_num + malignant_num, ), dtype='uint8')

    # store the malignant
    for i, malignant_file in enumerate(malignant_list):
        img = image.load_img(os.path.join(malignant_path, malignant_file), grayscale=False, target_size=(img_width,img_height))
    	_X[i] = image.img_to_array(img)
    # add the benign and set flag to 1 (this should be equal to "1D binary labels" as in the example flow_from_directory)
    for i, benign_file in enumerate(benign_list):
        img = image.load_img(os.path.join(benign_path, benign_file), grayscale=False, target_size=(img_width,img_height))
        _X[i + malignant_num] = image.img_to_array(img)
        _y[i + malignant_num] = 1
    return _X, _y

def save_bottleneck_features():
    """builds the pretrained vgg16 model and runs it on our training and validation datasets"""
    datagen = ImageDataGenerator(rescale=1./255)

    # match the vgg16 architecture so we can load the pretrained weights into this model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load VGG16 weights
    f = h5py.File(weights_path)

    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)

    f.close()
    print 'Model loaded.'

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


def train_top_model():
    """trains the classifier"""
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * nb_train_samples_benign + [1] * nb_train_samples_malignant)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * nb_validation_samples_benign + [1] * nb_validation_samples_maligant)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy','precision', 'recall'])

    scores = model.fit(train_data, train_labels,
                nb_epoch=nb_epoch,
                batch_size=32,
                validation_data=(validation_data, validation_labels),
                class_weight=class_weights)
                #callbacks=[early_stopping])

    f_hist_1.write(str(scores.history))
    f_hist_1.close()

    # save the model weights
    model.save_weights(top_model_weights_path)

def fine_tune():
    """recreates top model architecture/weights and fine tunes with image augmentation and optimizations"""
    X_test, y_test = load_data('valid')
    print('X_test shape:', X_test.shape)
    print(X_test.shape[0], 'test samples')
    X_test /= 255

    # reconstruct vgg16 model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load vgg16 weights
    f = h5py.File(weights_path)

    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)

    f.close()

    # add the classification layers
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy','precision', 'recall'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    # fine-tune the model
    #i = 1
    #for epoch in range(1,nb_epoch+1):
    scores = model.fit_generator(
                    train_generator,
                    samples_per_epoch=nb_train_samples,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples,
                    class_weight=class_weights)
                    #callbacks = save_best_only)

    #y_pred = model.predict_classes(X_test, batch_size=64)
    #file_name = 'y_pred_'+str(i)+'.txt'
    #np.savetxt(file_name, y_pred)

    #y_score = model.predict_proba(X_test, batch_size=64)
    #file_name = 'y_score_'+str(i)+'.txt'
    #np.savetxt(file_name, y_score)
    #i=i+1

    f_hist_2.write(str(scores.history))
    f_hist_2.close()

    # save the model
    json_string = model.to_json()

    with open('final_model_architecture.json', 'w') as f:
        f.write(json_string)

    model.save_weights('final_weights.h5')

    # Predictions
    predictions = model.predict_generator(validation_generator, nb_validation_samples)
    np.savetxt('y_pred.txt', predictions)

    return model

def prediction(model):
    #Load data as a Numpy array
    X_test, y_test = load_data('valid')
    print('X_test shape:', X_test.shape)
    print(X_test.shape[0], 'test samples')
    X_test /= 255

    y_pred = model.predict_classes(X_test, batch_size=64)
    np.savetxt('y_pred.txt', y_pred)

    y_score = model.predict_proba(X_test, batch_size=64)
    np.savetxt('y_score.txt', y_score)

def load_model():
    # load json and create model
    json_file = open('final_model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

if __name__ == "__main__":
    if(TRAIN):
        save_bottleneck_features()
        train_top_model()
        model = fine_tune()
        prediction(model)
    else:
        model = load_model()
        prediction(model)
