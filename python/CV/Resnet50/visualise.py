import matplotlib.pyplot as plt


def Visualise(list,title="loss",Save_path='python/CV/Resnet50/trainingLog/default.png'):

    for l in list:
        plt.plot(l)
        plt.xlabel('itterations')
        plt.ylabel('loss')
        plt.title(title)
    plt.savefig(Save_path)
    plt.show()
    

def PlotFromText(path:str):
    f = open(path,'r')
    lines = f.readlines()

    tl = []
    vl = []

    for line in lines:
        txt = line.split(':')
        if txt[0] == 'train_loss':
            tl.append(float(txt[1].strip()))
        elif txt[0] == 'test_loss':
            vl.append(float(txt[1].strip()))

    l = [tl,vl]
    return l

if __name__ == '__main__':
    PATH = 'python/CV/Resnet50/trainingLog/log0.txt'
    l = PlotFromText(PATH)
    Visualise(l)