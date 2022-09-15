import tkinter as tk

#klasse om het master window aan te maken
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid()
        self.grid_columnconfigure(1,minsize=200)
        self.grid_rowconfigure(2,minsize=50)
        self.patrenWindow = PatrenWindow(self)

    def FillFrame(self):
        tk.Label(self, text="sort patrens").grid(column=0,row=0)
        tk.Button(self,text="new patren window",command=self.NewPatrenWindow).grid(column=1,row=0)
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


    def GridOptions(self):
        tk.Button(self,text="back",command= lambda: [self.ClearFrame(),self.FillFrame()]).grid(column=0,row=0)

    

#klasse die het window voor de patronen aanmaakt, en de nodige functies om deze te tekenen oproept
class PatrenWindow():
    def __init__(self,master):
        self.window =  tk.Toplevel(master)
        self.window.title("patren window")

    def ChangePatren(self,patren):
        match patren:
            case 'line': print("line patren tonen")

            case 'grid': print("grid patren tonen")
            
            case default: print("zwart vlak")


#start applicatie window
mainWindow = Application()
mainWindow.FillFrame()
mainWindow.master.title("patren maker")
mainWindow.master.geometry("400x200")
mainWindow.mainloop()