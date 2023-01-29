import tkinter as tk
import cv2
import numpy as np
import makeFrames

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid()
        self.grid_columnconfigure(1,minsize=200)
        self.grid_rowconfigure(2,minsize=50)

    def FillFrame(self):
        tk.Button(self,text="good",command= lambda: [self.saveFrame('good'),self.nextFrame()]).grid(column=0,row=0)
        tk.Button(self,text="bad",command= lambda: [self.saveFrame('bad'),self.nextFrame()]).grid(column=1,row=0)
        tk.Button(self,text="skip",command= lambda: [self.nextFrame()]).grid(column=2,row=0)

    def nextFrame(self):
        img2_org = mf.getOvervieuwImage()
        placemap = np.zeros(img2_org.shape, dtype=np.uint8) * 255
        crop = mf.nextframe()
        img1 = crop[0]
        w,h = img1.shape[:2]
        dim = (h*4,w*4)
        resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
        placemap = changePos(placemap,crop[1],crop[2])
        cv2.imshow('image_segment',resized)
        overvieuw = cv2.addWeighted(placemap, 0.4, img2_org, 1 - 0.4, 0)
        cv2.imshow('image_overvieuw',overvieuw)
        cv2.waitKey(1)

    def saveFrame(self,kind):
        mf.saveframe(kind)

def changePos(map,ltc,rbc):
    map = cv2.rectangle(map, ltc, rbc, (150,50,150),3)
    return map

def MakeImageFrame():
    img2_org = mf.getOvervieuwImage()
    placemap = np.zeros(img2_org.shape, dtype=np.uint8) * 255
    crop = mf.nextframe()
    img1 = crop[0]
    w,h = img1.shape[:2]
    dim = (h*4,w*4)
    resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

    placemap = changePos(placemap,crop[1],crop[2])

    cv2.imshow('image_segment',resized)
    overvieuw = cv2.addWeighted(placemap, 0.4, img2_org, 1 - 0.4, 0)
    cv2.imshow('image_overvieuw',overvieuw)
    cv2.waitKey(1)

mf = makeFrames.makeFrames('python/Camera/out/picture69.jpg')
def main():
    MakeImageFrame()

    a = Application()
    a.FillFrame()
    a.master.title("pattern maker")
    a.master.geometry("400x100")
    a.mainloop()

main()