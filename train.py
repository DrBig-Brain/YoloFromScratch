import torch
import torchvision.transforms as transforms
import torch.optim as optim 
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils.intersection_over_union import IntersectionOverUnion
from utils.non_max_supression import NonMaxSuppression
from utils.mean_average_precision import MeanAveragePrecision
from utils.cellboxes_to_boxes import CellboxesToBoxes
from utils.get_bboxes import GetBboxes
from utils.plot_image import PlotImage
from utils.save_checkpoint import SaveCheckpoint
from utils.load_checkpoint import LoadCheckpoint
from loss import YoloLoss

seed = 123

torch.manual_seed(seed)

#HyperParameters etc

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ",DEVICE)
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img,bboxes 


transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave = True)
    mean_loss = []
    for batch_index, (x,y) in enumerate(loop):
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Update the progressbar

        loop.set_postfix(loss = loss.item())
        print(f"Mean Loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes = 2, num_classes =20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    if LOAD_MODEL:
        LoadCheckpoint(torch.load(LOAD_MODEL_FILE),model,optimizer)

    train_dataset = VOCDataset(
        "data/8examples.csv",
        transform=transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR
    )
    test_dataset = VOCDataset(
        "data/test.csv",
        transform=transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last =False,
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last =True,
    )

    for epoch in range(EPOCHS):
        #uncomment the below snippet to evaluate model performance visualy
        '''for x,y in train_loader:
            x = x.to(DEVICE)
            for idx in range(8):
                bboxes = CellboxesToBoxes(model(x))
                bboxes = NonMaxSuppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                PlotImage(x[idx].permute(1,2,0).to("cpu"),bboxes)

            import sys
            sys.exit()'''
        pred_boxes, target_boxes = GetBboxes(
            train_loader, model, iou_threshold= 0.5, threshold = 0.4
        )

        mean_avg_precision = MeanAveragePrecision(
            pred_boxes, target_boxes,iou_threshold=0.5,box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_precision} at EPOCH: {epoch}")

        if mean_avg_precision > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict()
            }
            SaveCheckpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)
        train_fn(train_loader,model,optimizer,loss_fn)

    checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict()
            }
    SaveCheckpoint(checkpoint, filename=LOAD_MODEL_FILE)


if __name__ == "__main__":
    main()
    