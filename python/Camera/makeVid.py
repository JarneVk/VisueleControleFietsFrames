import cv2,os
import numpy as np


pathOut = 'video_'+str(1000)+'.mp4'
fps = 30
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, (800,600))
stop = 0
for i, frame in enumerate(os.listdir('vid')):
    frame = cv2.imread(os.path.join('vid',frame))
    # cv2.imshow('prev',frame)
    # cv2.waitKey(1)
    out.write(frame)
    # cv2.imwrite('python/Camera/vid/'+str(stop)+'.jpg',frame)
    print(i)
out.release()
cv2.destroyAllWindows()