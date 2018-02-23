import numpy as np
np.random.seed(12321)  # for reproducibility
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.pooling import MaxPooling2D 
import h5py
from keras import backend as K
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

nb_classes = 10
num_classes = 10
img_rows, img_cols = 42, 28
nb_epoch = 3
batch_size = 64
K.set_image_dim_ordering('th')
# input image dimensions




def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    digit_input = Input(shape=(1, img_rows, img_cols))
	# A 2D Conv layer with 8 filters and relu activation
    x = Conv2D(8, 3, 3,  activation='relu')(digit_input)
	# A maxpooling layer with (2,2) size filter and (2,2) stried
    tower_3 = MaxPooling2D((2, 2), strides=(2, 2))(x)
	# A 2D conv layer with 16 filters and relu activation
    tower_4=Conv2D(16, 3, 3, activation='relu')(tower_3)
	# A maxpooling layer with (2,2) size filter and default, i.e. (1,1), stride
    tower_5 = MaxPooling2D((2, 2), strides=(1, 1))(tower_4)
	# Flattering then a Dense layer, and then dropout with rate 0.5
    tower_6 = Flatten()(tower_5)
    tower_7 = Dense(64, activation='relu')(tower_6)
    tower_7a = Dense(64, activation='relu')(tower_7)
    tower_8 = Dropout(0.5)(tower_7a)
	# Two outputs
    main_output = Dense(10, activation='softmax', name='main_output')(tower_8)
    auxiliary_output = Dense(10, activation='softmax', name='auxiliary_output')(tower_8)
    
    model = Model(input=[digit_input], output=[main_output, auxiliary_output])
    model.compile(loss='categorical_crossentropy',optimizer='adam',  metrics=['accuracy'], loss_weights=[0.5, 0.5])
	#==================== Fetch Data and Fit ===================#
    model.fit(X_train, [y_train[0], y_train[1]], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    objective_score = model.evaluate(X_test, [y_test[0], y_test[1]], batch_size=batch_size) # TO BE COMPLETED.
    print('Evaluation on test set:', dict(zip(model.metrics_names, objective_score)))
	
	#Uncomment the following line if you would like to save your trained model
	#model.save('./current_model_conv.h5')
    if K.backend()== 'tensorflow':
        K.clear_session()

if __name__ == '__main__':
	main()

