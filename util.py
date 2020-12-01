import json
import argparse
import os
import cv2


class DT_DNN_UTIL():
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        with open(annotation_dir) as f:
            self.json_file = json.load(f)

    def load_all_annotations(self):
         inputs = []
         predictions = [] # [[category,[bbox]],[category,[bbox]]]
         arr = os.listdir(self.image_dir)

         


    def return_single_annotation(self, image_name):
        image_annotation = self.json_file[image_name]
        bbox_count = 1
        for annotation in image_annotation:
            print("Category name is: {}".format((annotation)['cat_name']))
            print("Bounding box {}: coordinates: {}".format(
                bbox_count, (annotation)['bbox']))
            bbox_count += 1
        return image_annotation

    def return_single_image(self, image_name):
        try:
            image = cv2.imread(os.path.join(self.image_dir, image_name))
        except Exception as e:
            print(e)
            image = None
        return image

    def show_image_with_annotation(self, image_name):
        image = None
        image_annotation = self.return_single_annotation(image_name)
        image_base = self.return_single_image(image_name)
        bbox_count = len(image_annotation)
        for annotation in image_annotation:
            category = int(annotation['cat_id'])
            bbox = annotation['bbox']
            point1 = (int(bbox[0]), int(bbox[1]))
            point2 = (int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3]))
            if category == 1:
                colorKey = (255, 0, 0)
            elif category == 2:
                colorKey = (0, 255, 0)
            elif category == 3:
                colorKey = (0, 0, 255)
            else:
                colorKey = (255, 255, 255)
            image = cv2.rectangle(image_base, point1, point2,colorKey,2)
        cv2.imshow(image_name, image)
        cv2.waitKey()
        return image


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_name", required=True,
                    help="Identify what image annotation you want to load")
    ap.add_argument("--image_dir", required=True,
                    help="Identify where the image is")
    ap.add_argument("--annotation_dir", required=True,
                    help="Identify where the annotation is")
    args = vars(ap.parse_args())
    image_name = args["image_name"]
    image_dir = args["image_dir"]
    annotation_dir = args["annotation_dir"]
    node = DT_DNN_UTIL(image_dir, annotation_dir)
    node.show_image_with_annotation(image_name)
