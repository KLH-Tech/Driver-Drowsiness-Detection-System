import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augumentation

tf.test.is_gpu_available()

train_datagen= ImageDataGenerator(rescale=1./255, rotation_range=0.2,shear_range=0.2,
    zoom_range=0.2,width_shift_range=0.2,
    height_shift_range=0.2, validation_split=0.2)

train_data= train_datagen.flow_from_directory(r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Train',
                                target_size=(80,80),batch_size=8,class_mode='categorical',subset='training' )

validation_data= train_datagen.flow_from_directory(r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Train',
                                target_size=(80, 80), batch_size=8, class_mode='categorical',subset='validation')

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Test',
                                target_size=(80,80),batch_size=8,class_mode='categorical')

bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80,80,3)))
hmodel = bmodel.output
hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation='relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2,activation= 'softmax')(hmodel)
model = Model(inputs=bmodel.input, outputs= hmodel)
for layer in bmodel.layers:
    layer.trainable = False

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Model\model.h5',
                            monitor='val_loss',save_best_only=True,verbose=3)

earlystop = EarlyStopping(monitor = 'val_loss', patience=7, verbose= 3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )

callbacks=[checkpoint,earlystop,learning_rate]

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data,steps_per_epoch=train_data.samples//8,
                   validation_data=validation_data,
                   validation_steps=validation_data.samples//8,
                   callbacks=callbacks,
                    epochs=50)

#Model Evaluation
acc_vr, loss_vr = Model.evaluate_generator(validation_data)  #changed model.evaluate... to Model.evaluate
print(acc_vr)
print(loss_vr)

acc_tr, loss_tr = Model.evaluate_generator(train_data)
print(acc_tr)
print(loss_tr)

acc_test, loss_test = Model.evaluate_generator(test_data)
print(acc_test)
print(loss_test)