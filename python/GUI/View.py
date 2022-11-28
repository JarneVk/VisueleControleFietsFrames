import tkinter as tk
import os
import threading
from threading import Event
from PIL import ImageTk, Image
import cv2

class View():

    def __init__(self):
        #start applicatie window
        mainWindow = MainWindow()
        mainWindow.startPage()
        mainWindow.master.title("visuele controle van fietsframes")
        mainWindow.master.geometry("1000x400")
        mainWindow.mainloop()

class MainWindow(tk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.grid()
        self.grid_columnconfigure(1)
        self.grid_rowconfigure(2)

    def ClearFrame(self):
        for widget in self.winfo_children():
            try:
                widget.destroy()
            except:
                print('kon niet verwijderen')
            

    def startPage(self):
        tk.Label(self, text="Welkom").grid(column=1,row=0)
        tk.Button(self,text="testing",command= lambda: [self.ClearFrame(),threading.Thread(target=self.testingPage).start()]).grid(column=1,row=1)

        self.update()

    def testingPage(self):
        tk.Label(self,text='live camera feed').grid(column=1,row=1)
        f1 = tk.Frame()
        self.canvas= tk.Canvas(f1, width= 600, height= 400)
        self.canvas.pack(expand='YES', fill="both")
        self.getFrame() #initialise image
        f1.grid(column=1,row=2)
        ###test var###
        self.count = 2
        stopThread = False
        t1 = threading.Thread(target=self.getNextFrame, args=[stopThread])
        t1.start()

        tk.Button(text="take pictue").grid(column=2,row=3)
        
        tk.Button(self,text="back",command= lambda: [self.stopThread(),f1.destroy(),self.ClearFrame(),self.startPage()]).grid(column=0,row=0)

        self.update()

    def stopThread(boolean):
        boolean = True
    #####test#####
    def getNextFrame(self,event):
        while(True):
            if(self.count>29):
                self.count = 2

            im = cv2.imread('python/Camera/tmp_pict/picture'+str(self.count)+'.jpg')
            b,g,r = cv2.split(im)
            im = cv2.merge((r,g,b))
            img = Image.fromarray(im)
            resized_image= img.resize((400,300), Image.ANTIALIAS)
            self.image = ImageTk.PhotoImage(image = resized_image)
            self.count +=1
            self.canvas.delete('all')
            self.canvas.create_image(10,10, anchor='nw', image=self.image)

            self.update()
            if event:
                break

    def getFrame(self):

        im = cv2.imread('python/Camera/tmp_pict/picture1.jpg')
        b,g,r = cv2.split(im)
        im = cv2.merge((r,g,b))
        img = Image.fromarray(im)
        resized_image= img.resize((400,300), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(image = resized_image)

        self.canvas.delete('all')
        self.canvas.create_image(10,10, anchor='nw', image=self.image)
        
        
