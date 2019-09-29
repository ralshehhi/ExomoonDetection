
from sklearn.metrics import classification_report, confusion_matrix
import keras
import tensorflow as tf
import keras.backend as K

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def get_net(input_shape):

        inputs = Input(shape=input_shape)

        DROPOUT_RATE=0.3 #Dropout applied to last layer
        INNERLAYER_DROPOUT_RATE=0.3 #Dropout applied to inner layers

        x = BatchNormalization(axis=1, mode=0, name='bn_0_freq')(inputs)

        x = Convolution1D(64, 3, padding="same", name='conv1')(x)
        x = BatchNormalization(axis=1, name='bn1')(x)
        x = LeakyReLU(alpha=0.02)(x)
        x = MaxPooling1D(2, name='pool1')(x)
        x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout1')(x)

        x = Convolution1D(128, 3, padding="same", name='conv2')(x)
        x = BatchNormalization(axis=1, name='bn2')(x)
        x = LeakyReLU(alpha=0.02)(x)
        x = MaxPooling1D(2, name='pool2')(x)
        x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout2')(x)

        x = Convolution1D(128, 3, padding="same", name='conv3')(x)
        x = BatchNormalization(axis=1, name='bn3')(x)
        x = LeakyReLU(alpha=0.02)(x)
        x = MaxPooling1D(2, name='pool3')(x)
        x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout3')(x)

        x = Convolution1D(128, 3, padding="same", name='conv4')(x)
        
        x = Convolution1D(64, 3, padding="same", name='conv5')(x)
        x = BatchNormalization(axis=1, name='bn5')(x)
        x = LeakyReLU(alpha=0.02)(x)
        x = MaxPooling1D(4, name='pool5')(x)
        x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout5')(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs, x)
        
        return model
        
def get_lossFunction():

#############################################################################        
print("Preparing data...")
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=ts,random_state=42)
print('Training/Testing:', train_data.shape,train_label.shape,test_data.shape,test_label.shape)

print("Building network...")
model = get_net(input_shape)

print("Training...")
cp_file = 'model_epoch{epoch:02d}_acc{val_acc:.2f}.h5'
checkpoint = ModelCheckpoint(cp_file, monitor='val_loss', save_best_only=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
cb=[reduce_lr, early_stopping ,checkpoint]
model.fit(train_data,train_label,batch_size=bs,epochs=ep,validation_split=vs,callbacks=cb,verbose=0)

print("Predicting...")
pred_label = model.predict(test_data, batch_size=bs, verbose=0)

print('Evaluating...")
true_y = test_label
pred_y = pred_label
target_names = ['No Moon','Moon']
pred_label = np.round(pred_label)
pred_label = pred_label.astype(int)
print(classification_report(test_label, pred_label,target_names=target_names))

target_names = [0,1]
TN, FP, FN, TP = confusion_matrix(test_label, pred_label, labels=target_names).ravel()
TPR = float(TP)/float(TP+FN) #TP/P
TNR = float(TN)/float(TN+FP) #TN/N
FPR = float(FP)/float(FP+TN) #FP/N
FNR = float(FN)/float(TP+FN) #FN/P
print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
print('Sensitivity/TPR:', TPR)
print('Specificity/TNR:', TNR)
print('Fall-out/FPR:', FPR)


