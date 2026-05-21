import torch

def IntersectionOverUnion(boxes_preds, boxes_labels, box_format = "midpoint"):
    """
    Calculate intersection over union

    Parameters: 
        boxes_pred(tensor): predictions of Bounding Boxes (batch_size,4)
        boxes_labels(tensor): correct label of Bounding Boxes (batch_size,4)
        box_format (str): Correct label of Bounding Boxes (x,y,w,h) or (x1,y1,x2,y2)
    returns:
        tensor: Interseaction over union for all examples
    """


    # boxes_preds shape = (N,4) where n is batch size or number of samples
    # boxes_labels shape = (N,4) where n is batch size or number of samples
    if box_format == "midpoint":
        box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3]/2
        box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4]/2
        box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3]/2
        box1_y2 = boxes_preds[...,1:2] + boxes_preds[...,3:4]/2

        box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3]/2
        box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4]/2
        box2_x2 = boxes_labels[...,0:1] + boxes_labels[...,2:3]/2
        box2_y2 = boxes_labels[...,1:2] + boxes_labels[...,3:4]/2

    elif box_format == "corners":
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4]

        box2_x1 = boxes_labels[...,0:1]
        box2_y1 = boxes_labels[...,1:2]
        box2_x2 = boxes_labels[...,2:3]
        box2_y2 = boxes_labels[...,3:4]

    else:
        raise ValueError("box_format must be 'midpoint' or 'corners'")

    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1,box2_y1)
    x2 = torch.min(box1_x2,box2_x2)
    y2 = torch.min(box1_y2,box2_y2)

    # clamp handels the case when they donot intersect
    intersection = (x2 - x1).clamp(0)*(y2-y1).clamp()

    box1_area = (box1_x2-box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2-box2_x1) * (box2_y2 - box2_y1)

    union = box1_area + box2_area - intersection + 1e-6

    return intersection / union;
