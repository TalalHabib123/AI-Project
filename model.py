import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.applications import DenseNet201, Xception
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, GlobalMaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

RETRAIN = False

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

    X_train, X_test, y_train, y_test = train_test_split(data['image'], data['label'], test_size=0.2, random_state=random.randint(0, 1000))

    train_df = pd.DataFrame({'filename': X_train, 'class': y_train.astype('str')})
    test_df = pd.DataFrame({'filename': X_test, 'class': y_test.astype('str')})

    image_size = (256, 256)

    train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filename', y_col='class', target_size=image_size, class_mode='categorical')
    test_generator = test_datagen.flow_from_dataframe(test_df, x_col='filename', y_col='class', target_size=image_size, class_mode='categorical')
    prediction_generator = test_datagen.flow_from_dataframe(test_df, x_col='filename', target_size=image_size, class_mode=None, shuffle=False)

    if os.path.isfile('model.h5'):
        model = load_model('model.h5')
        
    else:
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

        for layer in base_model.layers:
            layer.trainable = False

        # x = base_model.output
        # x = GlobalMaxPooling2D()(x)
        # x = Flatten()(x)
        # x = Dense(2048, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(2048, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(2048, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(2048, activation='relu')(x)
        # # x = Dropout(0.5)(x)
        # x = Dense(2048, activation='relu')(x)
        # # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # # x = Dropout(0.5)(x)
        # # x = Dense(1024, activation='relu')(x)
        # # # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # # x = Dropout(0.25)(x)
        
        # x = base_model.output
        # x = MaxPooling2D()(x)
        # x =Conv2D(1024, (3, 3), activation='relu')(x)
        # x =MaxPooling2D()(x)
        # x =Dropout(0.3)(x)
        # x =Conv2D(1024, (3, 3), activation='relu')(x)
        # x =MaxPooling2D()(x)
        # x =Dropout(0.3)(x)
        # x =Conv2D(1024, (3, 3), activation='relu')(x)
        # x =MaxPooling2D()(x)
        # x =Dropout(0.3)(x)
        # x = Flatten()(x)
        # x = Dense(1024, activation='relu')(x)
        
        
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Flatten()(x)
        # x = Dense(2048, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(2048, activation='relu')(x)
        # x = Dense(2048, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(2048, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        
        # x = base_model.output
        # x = MaxPooling2D()(x)
        # x = Flatten()(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dense(1024, activation='relu')(x)
        # # x = Dropout(0.25)(x)
    
        predictions = Dense(7, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator, epochs=3, batch_size=64, verbose=1, validation_data=test_generator)
        
        model.save('model.h5')

    if RETRAIN:
        model.fit(train_generator, epochs=12, batch_size=64, verbose=1, validation_data=test_generator)
        model.save('model.h5')
        
    stimulate_predictions =[]
    for _ in range(1):
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
    
    df = pd.read_csv('predictions.csv')
     
    fig , ax = plt.subplots(2, 1, figsize=(10, 11))
    success_rate = df[df['Predicted Class'] == df['Actual Label']].groupby('Actual Label').size() / df.groupby('Actual Label').size()
    diseases = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    success_rate.index = success_rate.index.map(lambda x: diseases[int(x)])

    # Use a color palette and add the count above each bar
    sns.barplot(x=success_rate.index, y=success_rate.values, hue=success_rate.index, ax=ax[0], palette='viridis')
    for i, v in enumerate(success_rate.values):
        ax[0].text(i, v + 0.01, str(round(v, 2)), color='black', ha='center')
    ax[0].set_title('Probability of Successful Prediction for Each Label')
    ax[0].set_ylabel('Success Rate')
    ax[0].set_xlabel('Disease')
    
    total_count = df.groupby('Actual Label').size()
    successful_count = df[df['Predicted Class'] == df['Actual Label']].groupby('Actual Label').size()

    count_df = pd.DataFrame({'Total Count': total_count, 'Successful Prediction Count': successful_count}).reset_index()

    melted_df = count_df.melt(id_vars='Actual Label', var_name='Type', value_name='Count')
    melted_df['Actual Label'] = melted_df['Actual Label'].map(lambda x: diseases[int(x)])

    bar_plot = sns.barplot(x='Actual Label', y='Count', hue='Type', data=melted_df, palette='viridis', ax=ax[1])

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.1f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 10), 
                        textcoords = 'offset points')

    ax[1].set_title('Total Count and Successful Prediction Count for Each Disease')
    ax[1].set_ylabel('Count')
    ax[1].legend(frameon=True, title='Type', title_fontsize='13', loc='upper right')
    ax[1].set_xlabel('Disease')

    plt.tight_layout()
    plt.show()
    
    