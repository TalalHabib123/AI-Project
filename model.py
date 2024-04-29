import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.applications import DenseNet201, Xception
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    data = pd.read_csv('augmented_data.csv')
    data['image'] = data['image'].apply(lambda x: 'augmented_data/' + x + '.jpg')

    X_train, X_test, y_train, y_test = train_test_split(data['image'], data['label'], test_size=0.2, random_state=42)

    train_df = pd.DataFrame({'filename': X_train, 'class': y_train.astype('str')})
    test_df = pd.DataFrame({'filename': X_test, 'class': y_test.astype('str')})

    image_size = (400, 400)

    train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filename', y_col='class', target_size=image_size, class_mode='categorical')
    test_generator = test_datagen.flow_from_dataframe(test_df, x_col='filename', y_col='class', target_size=image_size, class_mode='categorical')
    prediction_generator = test_datagen.flow_from_dataframe(test_df, x_col='filename', target_size=image_size, class_mode=None, shuffle=False)

    if os.path.isfile('model.h5'):
        model = load_model('model.h5')
        
    else:
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(400, 400, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(2048, activation='swish')(x)
        # x = Dropout(0.5)(x)
        x = Dense(2048, activation='swish')(x)
        # x = Dropout(0.5)(x)
        x = Dense(2048, activation='swish')(x)
        x = Dropout(0.5)(x)
        x = Dense(2048, activation='swish')(x)
        # x = Dropout(0.5)(x)
        x = Dense(2048, activation='swish')(x)
        # x = Dropout(0.5)(x)
        x = Dense(2048, activation='swish')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='swish')(x)
        # x = Dropout(0.5)(x)
        x = Dense(1024, activation='swish')(x)
        # x = Dropout(0.5)(x)
        x = Dense(1024, activation='swish')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='swish')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='swish')(x)
        # x = Dropout(0.25)(x)
    
        predictions = Dense(7, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator, epochs=15, verbose=1, validation_data=test_generator)
        
        model.save('model.h5')

    stimulate_predictions =[]
    for _ in range(10):
        predictions = model.predict(prediction_generator, verbose=1)
        stimulate_predictions.append(predictions)
    predictions = np.mean(np.array(stimulate_predictions), axis=0)
    labels = np.argmax(predictions, axis=1)
    
    temp = pd.DataFrame(predictions, columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
    temp['Image'] = test_df['filename'].values
    temp['Predicted Class'] = labels
    temp['Actual Label']=test_df['class'].values
    df = temp[['Image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'Predicted Class', 'Actual Label']]
    
    df.to_csv('predictions.csv', index=False)
    
    accuracy = accuracy_score(y_test, labels)
    print(f'Accuracy: {accuracy}')