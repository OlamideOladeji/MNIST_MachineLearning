import numpy as np
import h5py
np.random.seed(12321)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True
K.set_image_dim_ordering('th')

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28
# input image dimensions

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    
    digit_input = Input(shape=(1, img_rows, img_cols))
    x = Flatten()(digit_input)
    x = Dense(128, activation='relu')(x)
    main_output = Dense(10, activation='softmax', name='main_output')(x)
    auxiliary_output = Dense(10, activation='softmax', name='auxiliary_output')(x)
    
    model = Model(input=digit_input, output=[main_output, auxiliary_output])
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',  metrics=['accuracy'], loss_weights=[0.5, 0.5])
    model.fit(X_train, [y_train[0], y_train[1]], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    objective_score = model.evaluate(X_test, [y_test[0], y_test[1]], batch_size=batch_size) #TO  BE COMPLETED.
    print('Evaluation on test set:', dict(zip(model.metrics_names, objective_score)))
	
	#Uncomment the following line if you would like to save your trained model
	#model.save('./current_model_conv.h5')
    if K.backend()== 'tensorflow':
        K.clear_session()

if __name__ == '__main__':
	main()


