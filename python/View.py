from asyncio.windows_events import NULL
from email.policy import default
import tkinter as tk

#klasse om het master window aan te maken
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid()
        self.patrenWindow = PatrenWindow(self)

    def FillFrame(self):
        tk.Label(self, text="sort patrens").grid(column=0,row=0)
        tk.Label(self, text="line").grid(column=0,row=1)
        tk.Button(self,text="show",command= lambda: self.patrenWindow.ChangePatren('line')).grid(column=1,row=1)


class PatrenWindow():
    def __init__(self,master):
        self.window =  tk.Toplevel(master)
        self.window.title("patren window")

    def ChangePatren(self,patren):
        match patren:
            case 'line': print("line patren tonen")
            
            case default: print("wit vlak")


#start applicatie window
mainWindow = Application()
mainWindow.FillFrame()
mainWindow.mainloop()