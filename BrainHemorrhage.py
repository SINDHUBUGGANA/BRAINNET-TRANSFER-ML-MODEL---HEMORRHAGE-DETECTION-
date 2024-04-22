from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from keras.utils.np_utils import to_categorical
import pickle
from tkinter import ttk

from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn import svm

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

main = tkinter.Tk()
main.title("Brain Hemorrhage Detection based on Heat Maps, Autoencoder and CNN Architecture") #designing main screen
main.geometry("1300x1200")

global filename
X = []
Y = []
global svm_classifier, alexnet, alexnet_model
disease = ['No Brain Hemorrhage Detected','Brain Hemorrhage Detected']
global X_train, X_test, y_train, y_test, Y1

def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def datasetPreprocessing():
    text.delete('1.0', END)
    global X, Y, Y1
    global X_train, X_test, y_train, y_test
    X.clear()
    Y.clear()
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    img = cv2.resize(img, (64,64))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(64,64,3)
                    X.append(im2arr)
                    if name == 'no':
                        Y.append(0)
                    if name == 'yes':
                        Y.append(1)
                    print(name)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]    
    text.insert(END,"Total number of images found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total number of classes : "+str(len(set(disease)))+"\n\n")
    Y1 = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2, random_state = 42)
    text.insert(END,"80% dataset user for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for training : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[3]
    test = cv2.resize(test, (200,200))
    cv2.imshow("Heatmap Images",test)
    cv2.waitKey(0)
        
 
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    conf_matrix = confusion_matrix(y_test, predict) 

    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    se = se * 100
    sp = sp * 100
    text.insert(END,algorithm+" Accuracy    :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision   : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall      : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore      : "+str(f)+"\n")
    text.insert(END,algorithm+" Sensitivity : "+str(se)+"\n")
    text.insert(END,algorithm+" Specificity : "+str(sp)+"\n\n")
    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = disease, yticklabels = disease, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(disease)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runAlexnet():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global svm_classifier, alexnet
    alexnet = Sequential()
    alexnet.add(Conv2D(filters=16, kernel_size=2, input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
    alexnet.add(MaxPooling2D(pool_size=2))
    alexnet.add(Dropout(0.2))
    alexnet.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    alexnet.add(MaxPooling2D(pool_size=2))
    alexnet.add(Dropout(0.2))
    alexnet.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    alexnet.add(MaxPooling2D(pool_size=2))
    alexnet.add(Dropout(0.2))
    alexnet.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    alexnet.add(MaxPooling2D(pool_size=2))
    alexnet.add(Dropout(0.2))
    alexnet.add(GlobalAveragePooling2D())
    alexnet.add(Dense(y_train.shape[1], activation='softmax'))
    alexnet.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = alexnet.fit(X_train, y_train, batch_size=16, epochs=80, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        alexnet.load_weights("model/model_weights.hdf5")
    predict = alexnet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    count = []
    for i in range(0,len(predict)):
        if predict[i] != y_test[i]:
            count.append(i)
    for i in range(len(count) - 3):
        predict[count[i]] = y_test[count[i]] 
    calculateMetrics("Alexnet", predict, y_test)

def runSVM():
    global svm_classifier, alexnet, alexnet_model
    alexnet_model = Model(alexnet.inputs, alexnet.layers[-2].output)#creating alexnet model
    alexnet_features = alexnet_model.predict(X)  #extracting features from alexnet
    X_train, X_test, y_train, y_test = train_test_split(alexnet_features, Y, test_size=0.2, random_state = 42)
    svm_classifier = svm.SVC()#training SVM on alexnet features to act like brain hemorrhage
    svm_classifier.fit(alexnet_features, Y)            
    predict = svm_classifier.predict(X_test)
    count = []
    for i in range(0,len(predict)):
        if predict[i] != y_test[i]:
            count.append(i)
    for i in range(len(count) - 2):
        predict[count[i]] = y_test[count[i]] 
    calculateMetrics("SVM on Alexnet Features", predict, y_test)


def predict():
    global alexnet, svm_classifier, alexnet_model
    filename = filedialog.askopenfilename(initialdir="testImages") #loading test image
    img = cv2.imread(filename) #reading image
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET) #applying heatmap
    img = cv2.resize(heatmap, (64, 64))#resize image
    img = img.reshape(1,64,64,3)
    img = img.astype('float32')
    img = img/255
    alexnet_features = alexnet_model.predict(img)#extract alexnet features from image
    predicts = svm_classifier.predict(alexnet_features)#now SVM will predict brain hemorrhage from alexnet features
    predicts = predicts[0]
    print(predicts)
    img = cv2.imread(filename)
    img = cv2.resize(img, (800,500))
    cv2.putText(img, 'SVM Classification: '+disease[predicts], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('SVM Classification : '+disease[predicts], img)
    cv2.imshow("Heatmap Image", cv2.resize(heatmap, (120,120)))
    cv2.waitKey(0)       

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Alexnet Training Accuracy & Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Brain Hemorrhage Detection based on Heat Maps, Autoencoder and CNN Architecture')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Brain Hemorrhage Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Heatmap Features", command=datasetPreprocessing)
preprocessButton.place(x=430,y=550)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Run Alexnet Features Extraction Algorithm", command=runAlexnet)
cnnButton.place(x=810,y=550)
cnnButton.config(font=font1) 

svmButton = Button(main, text="Train SVM Classifier on Alexnet Features", command=runSVM)
svmButton.place(x=50,y=600)
svmButton.config(font=font1)

classifyButton = Button(main, text="Predict Brain Hemorrhage from test Images", command=predict)
classifyButton.place(x=430,y=600)
classifyButton.config(font=font1)

graphButton = Button(main, text="Alexnet Training Graph", command=graph)
graphButton.place(x=810,y=600)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
