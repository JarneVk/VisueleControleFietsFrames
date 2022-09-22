from distutils.cmd import Command
from re import X
import tkinter as tk
from turtle import width
import Dimentions as Dm
import SqlQuerries

dimentions = Dm.Dimentions()
db = SqlQuerries.DataBase()

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

    def NewPatrenWindow(self):
        self.patrenWindow = PatrenWindow(self)

    def ClearFrame(self):
        for widget in self.winfo_children():
            try:
                widget.grid_forget()
            except:
                pass

    def LineOptions(self):
        tk.Button(self,text="back",command= lambda: [self.ClearFrame(),self.FillFrame()]).grid(column=0,row=0)
        tk.Label(self,text="width:").grid(column=0,row=1)
        self.LineEntry = tk.Entry(self)
        self.LineEntry.grid(column=1,row=1)
        tk.Button(self,text="apply", command= lambda: [self.readLines(),self.patrenWindow.drawLines()]).grid(column=2,row=1)
        tk.Button(self,text="save", command= lambda: self.saveLines()).grid(column=2,row=2)

        f1 = tk.Frame(self)
        tk.Button(f1,text="-1",command= lambda: [dimentions.subWidth(1),dimentions.subHeight(1), self.patrenWindow.drawLines()]).grid(column=0,row=0)
        tk.Button(f1,text="+1",command= lambda: [dimentions.addWidth(1),dimentions.addHeight(1),self.patrenWindow.drawLines()]).grid(column=1,row=0)
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
        tk.Button(self,text="apply", command= lambda: [self.readGrid(),self.patrenWindow.drawGrid()]).grid(column=2,row=2)
        tk.Button(self,text="save", command= lambda: self.saveGrid()).grid(column=2,row=3)

        f2 = tk.Frame(self)
        buttonList = tk.Text(f2)
        buttonList.pack(side="left")
        sb = tk.Scrollbar(f2,command=buttonList.yview)
        sb.pack(side="right")
        buttonList.configure(yscrollcommand=sb.set)
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
        match patren:
            case 'line': self.drawLines()
            

            case 'grid': self.drawGrid()
            
            case default: pass


    def toggleRot(self):
        if self.lineRotation:
            self.lineRotation = 0
        else:
            self.lineRotation = 1

    def drawLines(self):
        self.canvas.delete('all')
        width = dimentions.getWidth()
        height = dimentions.getHeight()
        if(self.lineRotation == 0):
            for i in range(0,int(dimentions.getHorizontal()),2):
                self.canvas.create_rectangle(width*i, 0, (width*i)+width, dimentions.resolution_y, outline="#000",fill="#000")
        else:
            for i in range(0,int(dimentions.getVertical()),2):
                self.canvas.create_rectangle(0,dimentions.height*i,dimentions.resolution_x,(dimentions.height*i)+dimentions.height, outline="#000",fill="#000")

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


#start applicatie window
mainWindow = Application()
mainWindow.FillFrame()
mainWindow.master.title("pattern maker")
mainWindow.master.geometry("1000x400")
mainWindow.mainloop()