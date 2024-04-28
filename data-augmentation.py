import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from PIL import Image
from tqdm import tqdm

datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

data=pd.read_csv('processed_data.csv')
newdata = pd.DataFrame(columns=['image', 'label'])

for index, row in tqdm(data.iterrows()):
    image = row['image']
    label = row['label']
    img = load_img('images/'+image+'.jpg')
    img = img.resize((224, 224))
    img.save('augmented_data/'+image+'.jpg')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    for i , batch in enumerate(datagen.flow(x, batch_size=1)):
        img = Image.fromarray(batch[0].astype('uint8'))
        img.save('augmented_data/'+image+'_'+str(i)+'.jpg')
        temp_data = pd.DataFrame({'image': [image+'_'+str(i)], 'label': [label]})
        newdata = pd.concat([newdata, temp_data], ignore_index=True)
        if i >= 5:
                break

data = pd.concat([data, newdata], ignore_index=True)

data.to_csv('augmented_data.csv', index=False)