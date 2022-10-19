from cgi import test
import sqlite3
from unicodedata import name

class DataBase():

    def __init__(self) :
        self.con = sqlite3.connect("VisueleControleFietsFrames.db")
        self.cur = self.con.cursor()

    def insertLine(self,w,name):
        try:
            width = int(w)
            data = [width,name]
            self.cur.execute("INSERT INTO LineSettings(width,name) VALUES(?, ?)",data)
            self.con.commit()
        except:
            print("insertLine error")

    def getLineSettings(self):
        try:
            self.cur.execute("SELECT * FROM LineSettings")
            return self.cur.fetchall()
        except:
            print("db error getLineSettings")

    def insertGrid(self,w,h,name):
        try:
            data = [int(w),int(h),name]
            self.cur.execute("INSERT INTO GridSettings(width,height,name) VALUES(?, ?,?)",data)
            self.con.commit()
        except:
            print("insertGrid error")

    def getGridSettings(self):
        try:
            self.cur.execute("SELECT * FROM GridSettings")
            return self.cur.fetchall()
        except:
            print("db error getGridSettings")