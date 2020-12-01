import tensorflow as tf
import cv2
import argparse
from util import DT_DNN_UTIL


class DT_OBSTACLE_Trainer():
    def __init__(self, image_dir, annotation_dir, epochs, batch, lr, split, model="CNN"):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.training = split
        self.model = model
        self.util = DT_DNN_UTIL(image_dir, annotation_dir)
        self.runner()
        exit(0)

    def runner(self):
        self.load_data()

    def load_data(self):
        dataset = self.util.load_all_annotations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter Setup")
    parser.add_argument("--image_dir", required=True,
                        help="The directory for image files")
    parser.add_argument("--annotation_dir", required=True,
                        help="The directory for annotation files")
    parser.add_argument(
        "--epochs", help="Set the total training epochs", default=1000
    )
    parser.add_argument("--batch", help="Set the batch size", default=64)
    parser.add_argument(
        "--lr", help="Set the initial learning rate", default=10e-3
    )
    parser.add_argument(
        "--split", help="percentage of training", default=10e-3
    )
    parser.add_argument(
        "--model", help="Which Neural Network Model to use", default="RCNN"
    )
    args = parser.parse_args()

    DT_OBSTACLE_Trainer(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        split=float(args.split),
        model=args.model
    )
