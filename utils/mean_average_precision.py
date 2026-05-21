import torch
from collections import Counter
from intersection_over_union import IntersectionOverUnion

def MeanAveragePrecision(
        pred_boxes,
        true_boxes,
        box_format,
        iou_threshold=0.5,
        num_classes = 20
        ):
    # pred_boxes: (list): [[train,class_pred,prob_score,x1,y1,x2,y2],...]
    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truth = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truth.append(true_box)

        #img 0 has 3 bboxes
        #img 1 has 5 bboxes
        #amount_bboxers = {0:3,1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truth])

        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)

        #amount_bboxes = {0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}
        detections.sort(key = lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))

        total_true_bboxes = len(ground_truth)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                    bbox for bbox in ground_truth if bbox[0] == detection[0]
                    ]

            num_ground_truth = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IntersectionOverUnion(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format = box_format,
                        )
                if iou.item() > best_iou:
                    best_iou = iou.item()
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] =1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]),recalls))
        average_precision.append(torch.trapz(precisions,recalls))

    return sum(average_precision) / len(average_precision)
