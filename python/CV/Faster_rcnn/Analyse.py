import numpy as np
import cv2
import torch
import glob as glob
from src.model import create_model

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=2).to(device)
model.load_state_dict(torch.load(
    'outputs/model5.pth', map_location=device
))
model.eval()
# classes: 0 index is reserved for background
CLASSES = [
    'background','bad'
]
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.7

def AnalyseFrame(image):
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        return orig_image
    return orig_image
    
def AnalyseVideo(path):
    video = cv2.VideoCapture(path)
    out = cv2.VideoWriter('test_predictions/video_out.mp4',cv2.VideoWriter_fourcc(*'MP4V'),15,(800,600))
    last_frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_id = 0
    while frame_id<last_frame_num:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        out_frame = AnalyseFrame(frame)
        out.write(out_frame)
        cv2.imshow('vid',out_frame)
        cv2.waitKey(1)
        frame_id+=1
    out.release()

    

if __name__ == '__main__':
    PATH = "video_1000.mp4"
    AnalyseVideo(PATH)