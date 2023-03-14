from Resnet50 import ResNet50
# from NetwerkPaper import PaperNet

##################################################################################
#
#                           CONFIG PARAMETERS
#
##################################################################################
EPOCHS = [20,   #1
          50,   #2
          100,  #3
          200,
          300,
          20,
          50,
          100,
          200,
          300]
LEARNING_RATES = [0.005,    #1
                  0.005,    #2
                  0.005,    #3
                  0.005,
                  0.005,
                  0.0005,
                  0.0005,
                  0.0005,
                  0.0005,
                  0.0005]

DATASET = 'dataset_2'

RESNET = True
PAPERNET = False

##################################################################################
#
#                           Benchmark
#
##################################################################################

class train_benchmark():

    def __init__(self) -> None:
        pass

    def trainLoop(self):
        for indx,epoch in enumerate(EPOCHS):
            try:
                lr = LEARNING_RATES[indx]
            except Exception:
                lr = 0.01
            
            if RESNET == True:
                self.Resnet50(epoch,lr)

            if PAPERNET == True:
                self.PaperNet()



    def Resnet50(self,epochs:int,lr:float):
        ResNet50.setLr(lr)
        #----------------------------------- creating dataset -------------------------------------------
        train_dataloader, test_dataloader,tsize,valsize,test_y,val_dataset,size_val = ResNet50.CreateDataset.LoadDataset(DATASET)
        #-----------------------------------   train model    -------------------------------------------
        epochs = int(epochs)
        resnet = ResNet50.Resnet50_CreateModel(train_dataloader, test_dataloader,val_dataset,epochs)
        resnet.train(tsize,valsize,size_val)

    def PaperNet(self,epochs:int,lr:float):
        pass





if __name__ == '__main__':
    tb=train_benchmark()
    tb.trainLoop()
    
