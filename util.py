import json
import argparse	
import os
import cv2

class dt_NN_util():
    def __init__(self,image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        with open(annotation_dir) as f:
            self.json_file = json.load(f)

    def return_single_annotation(self,image_name):
        image_annotation = self.json_file[image_name]
        bbox_count = 0
        for annotation in image_annotation:
    	    print("Category name is: {}".format((annotation)['cat_name']))
            print("Bounding box {}: coordinates: {}".format(bbox_count,(annotation)['bbox']))
            bbox_count+=1
        return image_annotation

    def return_single_image(self,image_name):
        try:
            image = cv2.imread(os.path.join(self.image_dir, image_name))
        except Exception as e:
            print(e)
        

    def show_image_with_annotation(self,image_name):
        image_annotation = self.json_file[image_name]
        bbox_count = 0
        for annotation in image_annotation:
    	    print("Category name is: {}".format((annotation)['cat_name']))
            print("Bounding box {}: coordinates: {}".format(bbox_count,(annotation)['bbox']))
            bbox_count+=1
        return image_annotation

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_name", required = True, help="Identify what image annotation you want to load")
    ap.add_argument("--image_dir", required = True, help="Identify where the image is")
    ap.add_argument("--annotation_dir",required = True,help="Identify where the annotation is")
    args = vars(ap.parse_args())
    image_name = args["imagename"]
    image_dir = args["image_dir"]
    annotation_dir = args["annotation_dir"]
