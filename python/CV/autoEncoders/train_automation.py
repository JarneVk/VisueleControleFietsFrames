import train

EPOCHS = 500

MODEL = 3 #1,2,3

DIR_PATH = "dataset_autoenc/good_dir"

channel_bases = [32,32,32,32 ,48,48,48,48     ,64,64,64,64]
latent_dim = [64,128,256,384 ,64,128,256,384  ,64,128,256,384]


train_dataloader, test_dataloader,tsize,valsize = train.LoadDataset(DIR_PATH)

for i in range(len(latent_dim)):
    model = train.trainAutoEncoder(train_dataloader,tsize,test_dataloader,int(EPOCHS),MODEL,base_channel_size=channel_bases[i],latent_dim=latent_dim[i])
    good,bad = train.validate(model,DIR_PATH,'dataset_autoenc/bad_dir')
