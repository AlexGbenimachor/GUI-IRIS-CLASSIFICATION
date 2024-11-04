#!/usr/bin/env python
import os
import csv
import json

try:
 import tkinter as tk #python3
except ImportError:
 import Tkinter as tk #python2
import numpy as np
#import tkFileDialog
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog
from perceptron_Dehydrate import *
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tkinter.scrolledtext import ScrolledText
from sklearn.model_selection import train_test_split
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk) 
  

'''
GUI INTERFACE PERCEPTRON INTERFACE
BY ALEX UGOCHUKWU GBENIMACHOR
'''


print("dehydration application initialization")


    
root = tk.Tk()
root.title("IRIS DATASET")
root.resizable(0,0)
#root.config(bg="teal")
root.geometry('800x600')
# Frame of fixed size
frame1 = tk.Frame(root, width=800, height=600)
frame1.grid(row=0, column=0)
frame1.config(bg="teal")
label1 = tk.Label(frame1, text="IRIS CLASSIFICATION", fg="green", font=("Arial", 12, "bold"))
label1.place(x=400, y=20, anchor=tk.CENTER)
fig = plt.figure(figsize = (2, 2), dpi = 100) 
canvas = FigureCanvasTkAgg(fig, master=frame1)
plot_widget = canvas.get_tk_widget()


FileNameText = StringVar()
DATASEtText = StringVar()

def create_dataset(xCsv):
    dataset =  list()
    for row in xCsv:
        dataset.append(row)
    dataset =  np.array(dataset[0:149], dtype=object)    
    label4.configure(text = str(dataset.shape))
    X_data =  np.array(dataset[:, 0:4], dtype = float)
    y_data =  np.array(dataset[:, -1], dtype = object)
    #print("y_data:", y_data)
    #yData =  np.array([0 if i=="Iris-setosa" 1 elif i=="Iris-versicolor"else i=2 for i in  y_data], dtype=float)
    y_TargetList =  list() 
    for i in y_data:
        if(i =="Iris-setosa"):
           i = 0
        elif(i=="Iris-versicolor"):
            i = 1
        else:
            i= 2
        y_TargetList.append(i)
    y_TargetList =  np.array(y_TargetList, dtype=int)
    #datasetX = np.array([X_data, y_TargetList])
    #print(datasetX)
    return np.array(X_data[0:100],dtype=float), np.array(y_TargetList[0:100] , dtype=int)
    
    #XY =  [ X_data, y_TargetList]
    
    
def plot(errors, window): 
  
    # the figure that will contain the plot 
    fig = Figure(figsize = (5, 5), 
                 dpi = 100) 
  
    # list of squares 
    y = [i**2 for i in range(101)] 
    L_errors = len(errors) 
    # adding the subplot 
    plot1 = fig.add_subplot(111) 
  
    # plotting the graph 
    plot1.plot(range(0, len(errors)),errors, marker="o") 
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                               master = window)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   window) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack() 
def Close(): 
    root.destroy() 
    os._exit(0)
    #canvas.destroy()
    #frame1.destroy()
def disable_event():
   pass  
  
# Button for closing 
exit_button = Button(frame1, text="Exit", command=Close) 
exit_button.place(x=100, y=500, anchor=tk.CENTER) 

def open_file():
    
    filename = tk.filedialog.askopenfilename(parent=frame1,title='Choose a file')
    FileNameText.set(filename)
    datafile =  open(filename)
    dataCsv =  csv.reader(datafile, delimiter=",")
    X_data, y_TargetList = create_dataset(dataCsv)
    #print(y_TargetList, X_data)
    x_train, x_test, y_train, y_test = train_test_split(X_data,y_TargetList, test_size=0.2, random_state=42)
    text_area.insert(INSERT,(X_data,y_TargetList))
    
    Ppn = perceptronX(learning_rate = 0.001, n_iter= 100)
    #Ppn =  Perceptron(random_state = 42, max_iter = 1000, tol = 0.001)
    Ppn.fit(x_train,y_train)
    errors =  Ppn.errors
    ypred =   Ppn.prediction(x_test) 
    accuracy = round(accuracy_score(ypred,  y_test),2) * 100
    label6.configure(text = str(accuracy))
   
    
    plt.plot(range(0, len(errors)),errors, marker="o")
    plt.draw()
#textbox...
label2 = tk.Label(frame1, text="BROWSE:", fg="green", font=("Arial", 12, "bold"))
label2.place(x=100, y=80, anchor=tk.CENTER)

# adding Entry Field
txt = Entry(frame1, textvariable=FileNameText, width=40)
txt.place(x=350, y=80, anchor=tk.CENTER)

btn1 = Button(frame1, text = "Browse" , fg = "black", command= open_file) 
btn1.place(x=570, y=80, anchor=tk.CENTER)

label3 = tk.Label(frame1, text="FILE SHAPE:", fg="green", font=("Arial", 12, "bold"))
label3.place(x=110, y=120, anchor=tk.CENTER)
label4 = tk.Label(frame1, text="", fg="black",bg="teal", font=("Arial", 12, "bold"))
label4.place(x=250, y=120, anchor=tk.CENTER)
# adding Entry Field
text_area = ScrolledText(frame1, width = 30,  height = 8,  font = ("Arial", 15, "bold")) 
text_area.place(x=350, y=250, anchor=tk.CENTER)
label5 = tk.Label(frame1, text="Accuracy:", fg="green", font=("Arial", 12, "bold"))
label5.place(x=100, y=400, anchor=tk.CENTER)
label6 = tk.Label(frame1, text="Accuracy", fg="black", bg="teal" , font=("Arial", 12, "bold"))
label6.place(x=200, y=400, anchor=tk.CENTER)
plot_widget = canvas.get_tk_widget()
plot_widget.place(x=350, y=480, anchor=tk.CENTER)
##
#
if __name__=="__main__":
   root.protocol("WM_DELETE_WINDOW", disable_event)
   frame1.mainloop()
