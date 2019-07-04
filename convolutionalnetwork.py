# Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# PART 1 Intialising the CNN
classifier = Sequential()

#  STEP1:CONVOLUTION
classifier.add(Conv2D(32,3, 3, input_shape = ( 64,64,3), activation = 'relu'))
# 32 is the no. of filter used
# 3 is the no of rows in the filter
# 43 is the no. of column in the filter


# STEP 2 POOLING
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# adding second covolutional layer
classifier.add(Conv2D(32,3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Size is taken as 2*2

#STEP 3 Flattening
classifier.add(Flatten()) 

#STEP 4 FULL CONNECTION
classifier.add(Dense(output_dim=128, activation = 'relu'))
classifier.add(Dense(output_dim=1, activation = 'sigmoid'))
#binary output that why sigmoid if more than 2 output we use softmax activation

# compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# PART 2 Fitting the CNN to the images
 
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2, 
                                   zoom_range = 0.2)
                                 #  horiontal_flip = True)
                                  
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
                                                 
                                                
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32, 
                                            class_mode = 'binary')
                                           
                                           
classifier.fit_generator(training_set, 
                         steps_per_epoch = 8000, 
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

#steps for epochs is the no. of images in the training data set                    
#prediction of first image of dog

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    