import os 
import torchvision.transforms as transforms 
import torch
import copy 

from PIL import Image, ImageDraw, ImageFont


def verify(image, boxes, labels, c):
    label_color_map = {k: c.distinct_colors[i] for i, k in enumerate(c.rev_label_map.keys())}
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        # Boxes
        box_location = boxes[i]
        draw.rectangle(xy=box_location, outline=label_color_map[labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
    image.show()

def verify2(image, boxes, labels, config, color, name="verify", plot_labels=True):
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        # Boxes
        box_location = boxes[i]
        draw.rectangle(xy=box_location, outline=color)
        draw.rectangle(xy=[l + 1. for l in box_location], outline=color)  # a second rectangle at an offset of 1 pixel to increase line thickness
    image.show()
    image.save(name + ".jpg")

def save_evaluations_image(image, boxes, labels, count, config, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + "pictures/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    trans = transforms.ToPILImage()
    image = trans(image.cpu())
    image.save(save_dir + str(count) + ".jpg")
    pos_image = copy.deepcopy(image)

    neg_samples = torch.where(labels[:, -1] == 1)[0]
    pos_samples = torch.where(labels[:, -1] == 0)[0]

    labels = labels.cpu()
    boxes = boxes.cpu()

    draw_pos = ImageDraw.Draw(pos_image)
    draw_neg = ImageDraw.Draw(image)

    for i in range(boxes.size(0)):
        box_location = boxes[i]
        draw = draw_neg if i in neg_samples else draw_pos
        color = 'red' if i in neg_samples else 'green'
        draw.rectangle(xy=box_location.numpy(), outline=color)
        draw.rectangle(xy=[l + 1. for l in box_location.numpy()], outline=color)  # a second rectangle at an offset of 1 pixel to increase line thickness

    print("----- Plot.py: Saving images ------")
    pos_image.save(save_dir + str(count) + "_pos" + ".jpg")
    image.save(save_dir + str(count) + "_neg" + ".jpg")
