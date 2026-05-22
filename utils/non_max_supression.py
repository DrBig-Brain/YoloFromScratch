import torch
from utils.intersection_over_union import IntersectionOverUnion

def NonMaxSuppression( predictions, iou_threshold, threshold, box_format="corners"):
    #prediction = list = [[class, probablity, x1, y1, x2, y2],]

    assert type(predictions) == list
    bboxes = [box for box in predictions if box[1] > threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
                box 
                for box in bboxes
                if box[0]!=chosen_box[0]
                or IntersectionOverUnion(
                    torch.tensor(chosen_box[2:]).unsqueeze(0),
                    torch.tensor(box[2:]).unsqueeze(0),
                    box_format = box_format,
                    ).item() < iou_threshold
                ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
