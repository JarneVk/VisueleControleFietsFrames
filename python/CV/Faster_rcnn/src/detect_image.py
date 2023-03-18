import numpy as np
import cv2
import torch
import glob as glob

import sys
sys.path.append('python/CV/Faster_rcnn/src')
from model import create_model

CLASSES = [
    'background','bad'
]


class detect_image():

    def __init__(self) -> None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load the model and the trained weights
        self.model = create_model(num_classes=2).to(device)
        self.model.load_state_dict(torch.load(
            'python/CV/Faster_rcnn/outputs/model5.pth', map_location=device
        ))
        self.model.eval()

        self.detection_threshold = 0.7

    def do_overlap(self,l1, r1, l2, r2):
        return not (r1.x < l2.x or l1.x > r2.x or r1.y < l2.y or l1.y > r2.y)

    def detect(self,image):
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
            outputs = self.model(image)

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        return_boxes = []
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= self.detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            draw_boxes = non_max_suppression_slow(draw_boxes, overlapThresh=0.3)
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):

                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                # cv2.putText(orig_image, pred_classes[j], 
                #             (int(box[0]), int(box[1]-5)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                #             2, lineType=cv2.LINE_AA)
                return_boxes.append(((int(box[0]), int(box[1])),
                                    (int(box[2]), int(box[3]))))

            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(500)
        
        return return_boxes


def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0,last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
        # return only the bounding boxes that were picked
    return boxes[pick]