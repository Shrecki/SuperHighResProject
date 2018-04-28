#Model Proposals

dropoutRate=0.2

#Model 1

model1=Sequential()

model1.add(Conv2D(27,3,input_shape=(33,33,3)))
model1.add(BatchNormalization())
model1.add(Dropout(dropoutRate))

model1.add(Conv2D(27,3))
model1.add(BatchNormalization())
model1.add(Dropout(dropoutRate))

model1.add(Conv2D(27,9))
model1.add(BatchNormalization())
model1.add(Dropout(dropoutRate))

model1.add(MaxPooling2D((3,3)))
model1.add(Reshape((21,21,3)))

model1.summary()
model1.compile(loss="mean_squared_error",optimizer="adam",metrics=[psnr])


#Model 2

model2=Sequential()
model2.add(Conv2D(27,3,input_shape=(33,33,3)))
model2.add(Dropout(dropoutRate))

model2.add(Conv2D(27,3))
model2.add(Dropout(dropoutRate))

model2.add(Conv2D(27,3))
model2.add(Dropout(dropoutRate))

model2.add(Conv2D(27,3))
model2.add(Dropout(dropoutRate))

model2.add(Conv2D(27,3))
model2.add(Dropout(dropoutRate))

model2.add(Conv2D(27,9))
model2.add(Dropout(dropoutRate))

model2.add(MaxPooling2D(2,2))

model2.add(Reshape((21,21,3)))

model2.summary()
model2.compile(loss="mean_squared_error",optimizer="adam",metrics=[psnr])

#Model 3

