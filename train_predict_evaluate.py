import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from functions import *

os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

K.set_image_data_format('channels_last')
CH_AXIS = -1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config = config)

def create_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_file', help='data file', type=str, default='data.npz')
	parser.add_argument('--label_file', help='label file', type=str, default='label.npz')
	parser.add_argument('--data_dir', help='directory', type=str, default='/data/')
	parser.add_argument('--result_dir', help='directory', type=str, default='/results/')

	parser.add_argument('--tsplit', help='precantage of splitting for testing data', type=float, default=0.30)
	parser.add_argument('--vsplit', help='precantage of splitting for validation data', type=float, default=0.30)

	parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=100)
	parser.add_argument('--batch_size', help='batch size', type=int, default=32)
	parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
	parser.add_argument('--opt', help= 'optimization for training', type=str, default='adamax')
	parser.add_argument('--loss', help= 'loss for training', type=str, default='mse')
	parser.add_argument('--weight', help='weight-loss', type=str, default='weight')
	
	args = parser.parse_args()
	
	return args
##########################################
args = create_args()
ep=args.num_epochs
bs=args.batch_size
lr = args.lr
opt = args.opt
loss = args.loss
ts = args.tsplit
vs = args.vsplit
w = args.weight

print("Reading files from...")
data = np.load(args.data_dir  + args.data_file)
label = np.load(args.data_dir + args.label_file)

print("Preparing data...")
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=ts,random_state=42)
train_data = train_data[:,:,np.newaxis]
test_data = test_data[:,:,np.newaxis]

print("Building network...")
input_shape = (train_data.shape[1],train_data.shape[2])
print(input_shape)
model = get_net(input_shape)

print("Training...")
cp_file = 'model_epoch{epoch:02d}_acc{val_acc:.2f}.h5'
check_folder = args.result_dir +'checkpoints'
if os.path.isdir(check_folder) == False:
	os.mkdir(check_folder)

cp_file = check_folder+'/model_epoch{epoch:02d}_acc{val_acc:.2f}.h5'	
log_file = check_folder+'/log.csv'
checkpoint = ModelCheckpoint(cp_file, monitor='val_loss', save_best_only=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
csv_logger = CSVLogger(log_file)
cb=[reduce_lr, early_stopping, csv_logger,checkpoint]
layer = model.layers[-3].output
model.compile(optimizer='Adam', loss=get_loss(layer,1), metrics = ['accuracy'])
model.fit(train_data,train_label,batch_size=bs,epochs=ep,validation_split=vs,callbacks=cb,verbose=0)

print("Predicting...")
pred_label = model.predict(test_data, batch_size=bs, verbose=0)

print("Evaluating...")
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



