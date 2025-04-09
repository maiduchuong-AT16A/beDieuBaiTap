from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imutils.video import VideoStream
import imutils
from scipy import misc
import cv2
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("output_dir: ",output_dir)
    name = 'lap'
    output_class_dir = os.path.join(output_dir, name)
    print("output_class_dir: ",output_class_dir)
    nrof_images_total = 1
    print(os.path.join(output_class_dir, name +'_'+ str(nrof_images_total) + '.png'))
    output_filename = os.path.join(output_class_dir, name +'_'+ str(nrof_images_total) + '.png')
    print(output_filename)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))