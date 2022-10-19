import sqlite3

con = sqlite3.connect("VisueleControleFietsFrames.db")
cur = con.cursor()

cur.execute("drop table LineSettings")
cur.execute("drop table GridSettings")


cur.execute("CREATE TABLE LineSettings(id INTEGER PRIMARY KEY AUTOINCREMENT, width int, name varchar(20))")

cur.execute("CREATE TABLE GridSettings(id INTEGER PRIMARY KEY AUTOINCREMENT, width int, height int , name varchar(20))")


