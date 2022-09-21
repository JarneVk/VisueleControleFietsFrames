import tkinter as tk
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
        print(lijst)
        for i in range(len(lijst)):
            textKnop = str(lijst[0]) + lijst[1]
            button = tk.Button(text=textKnop)
            buttonList.window_create("end",window=button)
            buttonList.insert("end","\n")
        buttonList.configure(state="disabled")
        f2.grid(column=3)


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
        if(self.lineRotation == 0):
            for i in range(0,int(dimentions.getHorizontal()),2):
                self.canvas.create_rectangle(dimentions.getWidth()*i, 0, (dimentions.getWidth()*i)+dimentions.getWidth(), dimentions.resolution_y, outline="#000",fill="#000")
        else:
            for i in range(0,int(dimentions.getVertical()),2):
                self.canvas.create_rectangle(0,dimentions.getHeight()*i,dimentions.resolution_x,(dimentions.getHeight()*i)+dimentions.getHeight(), outline="#000",fill="#000")

    def drawGrid(self):
        self.canvas.delete('all')
        offset = int(0)
        for i in range(0,int(dimentions.getHorizontal()),1):
            for j in range(0,int(dimentions.getVertical()),1):
                if((j+i)%2==0):
                    self.canvas.create_rectangle((dimentions.getWidth()*i), (dimentions.getHeight()*j),((dimentions.getWidth()*i)+dimentions.getWidth()),((dimentions.getHeight()*j)+dimentions.getHeight()), outline="#000",fill="#000")
                else:
                    pass #wit laten


#start applicatie window
mainWindow = Application()
mainWindow.FillFrame()
mainWindow.master.title("pattern maker")
mainWindow.master.geometry("400x200")
mainWindow.mainloop()