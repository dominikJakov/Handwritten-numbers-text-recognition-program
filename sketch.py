import matrix
import math
import random
import nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ast
from PIL import Image
from PIL import ImageGrab
import pyautogui


# Vraca broj koji NN misli da je tocan
def one_number(i):
    max_num = 0
    max_result = 0
   # print (i)
    for x in range(10):
        if i[x] > max_result:
            max_result = i[x]
            max_num = x
    return max_num

# Za odredivanje pokretanja "Train" i "Test"
load_train = False
load_test = False

# Ucitavanje Var za slike
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/mnist/"


print ("Start load") # Pocetak load

# Loading Train
if input("load train data: T / F : ").upper() == "T":
    print ("Loading Train data...")
    load_train = True
    train_data = np.loadtxt("mnist_train_new.csv", delimiter=",")

# Loading Test
if input("load test data: T / F : ").upper() == "T":
    print ("Loading Test data...")
    load_test = True
    test_data = np.loadtxt("mnist_test.csv", delimiter=",")
print ("Finish Load")

fac = 255  * 0.99 + 0.01 # Smanjivanje vrijednosti piksela od 0 - 1

# Pravljenje liste za train
if load_train == True:
    train_imgs = np.asfarray(train_data[:, 1:]) / fac
    train_labels = np.asfarray(train_data[:, :1])

# Pravljenje liste za test
if load_test == True:
    test_imgs = np.asfarray(test_data[:, 1:]) / fac
    test_labels = np.asfarray(test_data[:, :1])



lr = np.arange(1)
for label in range(10):
    one_hot = (lr==label).astype(np.int)
    #print("label: ", label, " in one-hot representation: ", one_hot)
lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
if load_train == True:
    train_labels_one_hot = (lr==train_labels).astype(np.float)
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99

if load_test == True:
    test_labels_one_hot = (lr==test_labels).astype(np.float)
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99





# ||||||||||||||||||||||STVARANJE NEURAL NETWORK||||||||||||||||||||||||||||||

n_n = nn.Neurol_Network(image_pixels,100,10) # Nas Neural Network

# Otvaranje podata za weights i bias
f = open("output.txt", "r")
n_n.weights_ih.data = ast.literal_eval(f.readline())
n_n.weights_ho.data = ast.literal_eval(f.readline())

n_n.bias_h.data  = ast.literal_eval(f.readline())
n_n.bias_o.data  = ast.literal_eval(f.readline())


# Funkicja na traing
if load_train == True:
    print ("Start Traning...")
    for i in range(len(train_imgs)):
        n_n.train(train_imgs[i], train_labels_one_hot[i])
    print ("Finish Traning")

tocnih = 0 # Broj tocnih pogodaka
netocnih = 0 # Broj netocnih pogodaka

# Funkcija za testiranje
if load_test == True:
    print ("Start Testing...")
    d = input ("How many do you wanna test: ")
    for i in range(int(d)):
        res = n_n.feedfoward(test_imgs[i])
        number = one_number(res)
        print(test_labels[i],"number: ", number)
        if str(test_labels[i])[1:2] == str(number):
            tocnih += 1
        else:
            netocnih+= 1
            #img = test_imgs[i].reshape((28,28))
            #plt.imshow(img, cmap="Greys")
            #plt.show()

# Prinatnje rezultata testiranja
if load_test == True:
    print ("In total: ", tocnih + netocnih)
    print ("Correct: ", tocnih)
    print ("Incorrect: ", netocnih)
    print ("Correct - Average: ", tocnih / (tocnih + netocnih))
    print ("Incorrect - Average: ", netocnih / (tocnih + netocnih))
    print ("")

# U slucaju novog testiranja unosenje vrijedonsti u file(Za kasnije ucitavanje)
if load_train == True:
    n_n_i_l  = n_n.input_nodes
    n_n_h_l  = n_n.hidden_nodes
    n_n_o_l  = n_n.output_nodes

    n_n_w_ih = n_n.weights_ih.data
    n_n_w_ho = n_n.weights_ho.data

    n_n_b_h  = n_n.bias_h.data
    n_n_b_o  = n_n.bias_o.data

    f = open("output.txt", "w")

    f.write(str(n_n_w_ih) + "\n")
    f.write(str(n_n_w_ho) + "\n")
    f.write(str(n_n_b_h) + "\n")
    f.write(str(n_n_b_o) + "\n")
    f.close()

def rbg_to_one(rgb_list):
    return 1 - (((rgb_list[0] / 255) + (rgb_list[1] / 255) + (rgb_list[2] / 255)) / 3)
    
if input("Test free images: ").upper() == "T":
    while True:
        px=ImageGrab.grab().load()
        data = []
        guess = 0
        im = Image.new("RGB", (28,28))
        for y in range(152,180):
            for x in range(6,34):
                data.append(rbg_to_one(px[x,y]))
        guess = one_number(n_n.feedfoward(data))
        print (guess)
    

print ("Testing Finished")
