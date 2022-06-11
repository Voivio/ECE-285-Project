import os.path

import numpy as np
import cv2 as cv

from path import Path
from imageio import imread
import argparse

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm


def save_each_frame(i, img, depth, output_dir):
    img_array = imread(img)

    fig, axes = plt.subplots(nrows=2, figsize=(6.5, 4 + 0.5))

    axes[0].imshow(img_array)
    axes[0].set_axis_off()
    axes[1].imshow(1 / depth)
    axes[1].set_axis_off()

    fig.savefig(os.path.join(output_dir, "{}.png".format(i)), bbox_inches='tight')
    plt.close(fig)


def write_video(args):
    # read the prediction
    pred = np.load(args.prediction_path)
    # h, w = pred[0].shape
    h = 364
    w = 523
    print(f"height: {h}, width: {w}")

    output_dir = os.path.dirname(args.output_path)
    print("saving to {}".format(output_dir))

    scene_folder = Path(args.dataset_path)
    imgs = sorted(scene_folder.files('*.jpg'))

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(args.output_path, fourcc, 20.0, (w, h))

    for i, img in tqdm(enumerate(imgs)):
        save_each_frame(i, img, pred[i], output_dir)
        frame = cv.imread(os.path.join(output_dir, "{}.png".format(i)))
        out.write(frame)

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep VO with Sequential Learning Optimization')

    parser.add_argument('--dataset_path', type=str, default='',
                        help='path to input images')
    parser.add_argument('--prediction_path', type=str, default='/content/output/predictions.npy',
                        help='path to predictions.npy')
    parser.add_argument('--output_path', type=str, default='/content/output/depth.avi', help='path to output file')

    args = parser.parse_args()
    write_video(args)
