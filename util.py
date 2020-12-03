from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import json
import argparse
import os
import io
import cv2
import tensorflow as tf
from PIL import Image
import sys
sys.path.append('./')
from object_detection.utils import dataset_util


class DT_DNN_UTIL():
    def __init__(self, image_dir, annotation_dir, output_dir='train.record'):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        with open(annotation_dir) as f:
            self.json_file = json.load(f)
        self.tf_record_output = output_dir
        self.counted_images = 0
        self.skipped_images = 0

    def create_tf_record(self):
        writer = tf.io.TFRecordWriter(self.tf_record_output)
        image_name_list = os.listdir(self.image_dir)
        for image_name in image_name_list:
            tf_example = self.create_tf_example(image_name)
            if tf_example != None:
                writer.write(tf_example.SerializeToString())
                print("Image {} done...".format(image_name))
            else:
                continue
        writer.close()
        print("Record Creation Successfully! Skipped {} images... Created {} Images".format(
            self.skipped_images, self.counted_images))

    def create_tf_example(self, image_id):
        image_format = b'png'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        try:
            encoded_img, width, height = self.encode_single_image(image_id)
        except Exception as e:
            print("Image {} has {} issue... Skipping...".format(image_id, e))
            self.skipped_images += 1
            return
        annotations = self.return_single_annotation(image_name)
        filename = image_name.encode('utf8')
        image_format = b'png'
        for bbox in annotations:
            unannotated_bbox =(bbox)['bbox']
            point1 = (int(unannotated_bbox[0]), int(unannotated_bbox[1]))
            point2 = (int(unannotated_bbox[0])+int(unannotated_bbox[2]), int(unannotated_bbox[1])+int(unannotated_bbox[3]))
            xmins.append(point1[0] / width)
            xmaxs.append(point2[0] / width)
            ymins.append(point1[1] / height)
            ymaxs.append(point2[0] / height)
            classes_text.append(bbox['cat_name'].encode('utf8'))
            classes.append(bbox['cat_id'])
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_img),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def return_single_annotation(self, image_name):
        image_annotation = self.json_file[image_name]
        bbox_count = 1
        for annotation in image_annotation:
            #print("Category name is: {}".format((annotation)['cat_name']))
            #print("Bounding box {}: coordinates: {}".format(
            #    bbox_count, (annotation)['bbox']))
            bbox_count += 1
        return image_annotation

    def encode_single_image(self, image_name):
        image_path = os.path.join(self.image_dir, image_name)
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
        return encoded_jpg, width, height

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
            image = cv2.rectangle(image_base, point1, point2, colorKey, 2)
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
    node.create_tf_record()