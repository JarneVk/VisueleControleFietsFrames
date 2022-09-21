class Dimentions():
    def __init__(self):
        self.Widt = 10
        self.Height = 10

        self.resolution_x = 1920
        self.resolution_y = 1080

    #  Width   ##################################
    def setWidth(self,h):
        try:
            self.Widt = int(h)
        except:
            print("geen int")

    def addWidth(self,a):
        self.Widt += a
    
    def subWidth(self,a):
        self.Widt -= a
    
    def getWidth(self):
        return self.Widt

    def getHorizontal(self):
        count = self.resolution_x/self.Widt
        return count

    # Height ####################################
    def setHeight(self,h):
        try:
            self.Height = int(h)
        except:
            print("geen int")

    def addHeight(self,a):
        self.Height += a
    
    def subHeight(self,a):
        self.Height -= a
    
    def getHeight(self):
        return self.Height

    def getVertical(self):
        count = self.resolution_y/self.Height
        return count

    