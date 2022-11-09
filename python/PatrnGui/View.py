from time import sleep
import tkinter as tk
import os
from PatrnGui import Dimentions as Dm
from PatrnGui import SqlQuerries
import threading


dimentions = Dm.Dimentions()
db = SqlQuerries.DataBase()

class View():

    def __init__(self):
        #start applicatie window
        mainWindow = Application()
        mainWindow.FillFrame()
        mainWindow.master.title("pattern maker")
        mainWindow.master.geometry("1000x400")
        mainWindow.mainloop()

#klasse om het master window aan te maken
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid()
        self.grid_columnconfigure(1,minsize=200)
        self.grid_rowconfigure(2,minsize=50)
        self.patrenWindow = PatrenWindow(self)

    def FillFrame(self):
        tk.Label(self, text="sort patterns").grid(column=0,row=0)
        tk.Button(self,text="new pattern window",command=self.NewPatrenWindow).grid(column=1,row=0)
        tk.Label(self, text="line").grid(column=0,row=1)
        tk.Button(self,text="show",command= lambda: [self.ClearFrame(),self.LineOptions() ,self.patrenWindow.ChangePatren('line')]).grid(column=1,row=1)
        tk.Label(self, text="grid").grid(column=0,row=2)
        tk.Button(self,text="show",command= lambda: [self.ClearFrame(),self.GridOptions() ,self.patrenWindow.ChangePatren('grid')]).grid(column=1,row=2)
        tk.Label(self, text="presets").grid(column=0,row=3)
        tk.Button(self,text="show",command= lambda: [self.ClearFrame(),self.presetOptions()]).grid(column=1,row=3)

    def NewPatrenWindow(self):
        self.patrenWindow = PatrenWindow(self)

    def ClearFrame(self):
        for widget in self.winfo_children():
            try:
                widget.grid_forget()
            except:
                pass

    def setEntrytext(self,text):
        self.LineEntry.delete(0,tk.END)
        self.LineEntry.insert(0,text)

    def LineOptions(self):
        tk.Button(self,text="back",command= lambda: [self.ClearFrame(),self.FillFrame()]).grid(column=0,row=0)
        tk.Label(self,text="width:").grid(column=0,row=1)
        self.LineEntry = tk.Entry(self)
        self.LineEntry.grid(column=1,row=1)
        tk.Button(self,text="apply", command= lambda: [self.readLines(),threading.Thread(target=self.patrenWindow.drawLines).start()]).grid(column=2,row=1)
        tk.Button(self,text="save", command= lambda: self.saveLines()).grid(column=2,row=2)

        f1 = tk.Frame(self)
        tk.Button(f1,text="-1",command= lambda: [dimentions.subWidth(1),dimentions.subHeight(1), self.patrenWindow.drawLines(),self.setEntrytext(dimentions.getWidth())]).grid(column=0,row=0)
        tk.Button(f1,text="+1",command= lambda: [dimentions.addWidth(1),dimentions.addHeight(1),self.patrenWindow.drawLines(),self.setEntrytext(dimentions.getWidth())]).grid(column=1,row=0)
        f1.grid(column=1,row=2)

        tk.Button(self,text="fullscreen",command= lambda: self.patrenWindow.Fullscreen()).grid(column=2,row=0)
        tk.Button(self,text="rotate", command= lambda: [self.patrenWindow.toggleRot(),self.patrenWindow.drawLines()]).grid(column=1,row=3)

        f2 = tk.Frame(self)
        buttonList = tk.Text(f2)
        buttonList.pack(side="left")
        sb = tk.Scrollbar(f2,command=buttonList.yview)
        sb.pack(side="right")
        buttonList.configure(yscrollcommand=sb.set)
        lijst = db.getLineSettings()
        try:
            widthLijst = [i[1] for i in lijst]
            nameLijst = [i[2] for i in lijst]
            btn = []
            for i in range(len(lijst)):
                textKnop = str(widthLijst[i])+"|" + nameLijst[i]
                btn.append(tk.Button(text=textKnop, command=lambda c=i: self.processSavedButtonLine(btn[c].cget("text"))))
                buttonList.window_create("end",window=btn[i])
                buttonList.insert("end","\n")
            buttonList.configure(state="disabled")
            f2.grid(column=3,row=0,rowspan=4)
        except TypeError as e:
            pass

    def processSavedButtonLine(self,text):
        lijst = text.split("|")
        dimentions.setWidth(lijst[0])
        dimentions.setHeight(lijst[0])
        self.patrenWindow.drawLines()

    def readLines(self):
        dimentions.setWidth(self.LineEntry.get())
        dimentions.setHeight(self.LineEntry.get())

    def saveLines(self):
        data = [self.LineEntry.get(),0]
        self.SaveWindow(data,'line')

    def GridOptions(self):
        tk.Button(self,text="back",command= lambda: [self.ClearFrame(),self.FillFrame()]).grid(column=0,row=0)
        tk.Button(self,text="fullscreen",command= lambda: self.patrenWindow.Fullscreen()).grid(column=2,row=0)

        tk.Label(self,text="width:").grid(column=0,row=1)
        self.GridEntry_x = tk.Entry(self)
        self.GridEntry_x.grid(column=1,row=1)

        tk.Label(self,text="Height:").grid(column=0,row=2)
        self.GridEntry_y = tk.Entry(self)
        self.GridEntry_y.grid(column=1,row=2)
        tk.Button(self,text="apply", command= lambda: [self.readGrid(),threading.Thread(target=self.patrenWindow.drawGrid).start()]).grid(column=2,row=2)
        tk.Button(self,text="save", command= lambda: self.saveGrid()).grid(column=2,row=3)

        f2 = tk.Frame(self)
        buttonList = tk.Text(f2)
        buttonList.pack(side="left")
        sb = tk.Scrollbar(f2,command=buttonList.yview)
        sb.pack(side="right")
        buttonList.configure(yscrollcommand=sb.set)
        try:
            lijst = db.getGridSettings()
            widthLijst = [i[1] for i in lijst]
            heightLijst = [i[2] for i in lijst]
            nameLijst = [i[3] for i in lijst]
            btn = []
            for i in range(len(lijst)):
                textKnop = str(widthLijst[i])+"|"+str(heightLijst[i])+"|" + nameLijst[i]
                btn.append(tk.Button(text=textKnop, command=lambda c=i: self.processSavedButtonGrid(btn[c].cget("text"))))
                buttonList.window_create("end",window=btn[i])
                buttonList.insert("end","\n")
            buttonList.configure(state="disabled")
        except TypeError as e:
            pass
        f2.grid(column=3,row=0,rowspan=4)

    def processSavedButtonGrid(self,text):
        lijst = text.split("|")
        dimentions.setWidth(lijst[0])
        dimentions.setHeight(lijst[1])
        self.patrenWindow.drawGrid()

    def readGrid(self):
        dimentions.setWidth(self.GridEntry_x.get())
        dimentions.setHeight(self.GridEntry_y.get())

    def saveGrid(self):
        data = [self.GridEntry_x.get(),self.GridEntry_y.get()]
        self.SaveWindow(data,'grid')


    def presetOptions(self):
        tk.Button(self,text="back",command= lambda: [self.ClearFrame(),self.FillFrame()]).grid(column=0,row=0)
        tk.Button(self,text="fullscreen",command= lambda: self.patrenWindow.Fullscreen()).grid(column=2,row=0)

        f1 = tk.Frame(self)
        try:
            line = db.getLineSettings()
            widthLine = [i[1] for i in line]
            nameLine = [i[2] for i in line]
            grid = db.getGridSettings()
            widthGrid = [i[1] for i in grid]
            heightGrid = [i[2] for i in grid]
            nameGrid = [i[3] for i in grid]

            tk.Label(f1,text="____Line presets____").grid(row=0,column=0)
            grid_pos = 1
            for i in range(len(line)):
                tekst = str(widthLine[i]) + " | "+ str(nameLine[i])
                tk.Label(f1,text=tekst).grid(row=grid_pos,column=0)
                grid_pos += 1

            tk.Label(f1,text="____Grid presets____").grid(row=grid_pos,column=0)
            grid_pos += 1

            for i in range(len(grid)):
                tekst = str(widthGrid[i]) + " | "+ str(heightGrid[i])+" | "+str(nameGrid[i])
                tk.Label(f1,text=tekst).grid(row=grid_pos,column=0)
                grid_pos += 1

            f1.grid(column=1,row=1)
        except TypeError as e:
            pass

        tk.Button(self,text="cycle presets",command= lambda: threading.Thread(target=self.cyclePresets, args=(widthLine,widthGrid,heightGrid)).start()).grid(column=1,row=2)

#####################cycle presets ###########################################
    def cyclePresets(self,widthLine,widthGrid,heightGrid):
        for i in range(len(widthLine)):
            dimentions.setHeight(widthLine[i])
            dimentions.setWidth(widthLine[i])
            self.patrenWindow.drawLines()
            sleep(5)

        for i in range(len(widthGrid)):
            dimentions.setHeight(heightGrid[i])
            dimentions.setWidth(widthGrid[i])
            self.patrenWindow.drawGrid()
            sleep(5)        

#############################################################################

    def SaveWindow(self,data,kind):
        self.sWindow = tk.Toplevel(self)
        self.sWindow.title("save")
        self.sWindow.grid()

        tk.Label(self.sWindow,text="name:").grid(column=0,row=0)
        self.sNameEntery = tk.Entry(self.sWindow)
        self.sNameEntery.grid(column=1,row=0)
        f1 = tk.Frame(self.sWindow)
        tk.Button(f1,text="save",command= lambda: self.SaveDB(data,kind)).grid(column=1,row=0)
        tk.Button(f1,text="cancel",command= lambda: self.SaveDestroy()).grid(column=0,row=0)
        f1.grid(column=1,row=1)

    def SaveDB(self,data,kind):
        if kind == 'line':
            db.insertLine(data[0],self.sNameEntery.get())
        if kind == 'grid':
            db.insertGrid(data[0],data[1],self.sNameEntery.get())

        self.SaveDestroy()

    def SaveDestroy(self):
        self.sWindow.destroy()
        self.sWindow.update()


    

#klasse die het window voor de patronen aanmaakt, en de nodige functies om deze te tekenen oproept
class PatrenWindow():
    def __init__(self,master):
        self.window =  tk.Toplevel(master)
        self.window.title("pattern window")
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(fill=tk.BOTH, expand=True) #zorgt dat canvas fullscreen is

        self.lineRotation = 0

    def Fullscreen(self):
        self.window.attributes("-fullscreen",True)

    def ChangePatren(self,patren):
        if(patren=='line'):
            self.drawLines()
        elif(patren=='grid'):
            self.drawGrid()


    def toggleRot(self):
        if self.lineRotation:
            self.lineRotation = 0
        else:
            self.lineRotation = 1

    def drawLines(self):
        print("teken lines")
        self.canvas.delete('all')
        width = dimentions.getWidth()
        height = dimentions.getHeight()
        if(self.lineRotation == 0):
            for i in range(0,int(dimentions.getHorizontal()),2):
                self.canvas.create_rectangle(width*i, 0, (width*i)+width, dimentions.resolution_y, outline="#000",fill="#000")
        else:
            for i in range(0,int(dimentions.getVertical()),2):
                self.canvas.create_rectangle(0,height*i,dimentions.resolution_x,(height*i)+height, outline="#000",fill="#000")
        self.canvas.update()

    def drawGrid(self):
        self.canvas.delete('all')
        width = dimentions.getWidth()
        height = dimentions.getHeight()
        for i in range(0,int(dimentions.getHorizontal()),1):
            for j in range(0,int(dimentions.getVertical()),1):
                if((j+i)%2==0):
                    self.canvas.create_rectangle((width*i), (height*j),((width*i)+width),((height*j)+height), outline="#000",fill="#000")
                else:
                    pass #wit laten
        self.canvas.update()
            