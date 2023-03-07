import os
import pickle
import numpy as np
import nltk
from PIL import Image

from collections import defaultdict
import json


import torch
import torch.utils.data as data

device = "cuda" if torch.cuda.is_available() else "cpu"

def createIndex(json_file):
    f = open(json_file)
    file = json.load(f)
    anns = {}
    imgToAnns = defaultdict(list)
    if 'annotations' in file:
        for ann in file['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann
    return anns

def loadImg(ids, json_file):
    f = open(json_file)
    file = json.load(f)
    for i in range(len(file['images'])):
        if file['images'][i]['id'] == ids:
            image_name = file['images'][i]['file_name']
            return image_name

class VizWizDataset(data.Dataset):

    def __init__(self, root, json, vocab, transform=None):

        self.root = root
        self.vocab = vocab
        self.json = json
        self.transform = transform
        self.vizwiz = createIndex(json)
        self.ids = list(self.vizwiz.keys())

    def __getitem__(self, index):

        json = self.json
        vizwiz = self.vizwiz
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = vizwiz[ann_id]['caption']
        img_id = vizwiz[ann_id]['image_id']
        path = loadImg(img_id, json)

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target

    def __len__(self):
        f = open(self.json)
        file = json.load(f)
        return len(file['images'])

def collate_fn(data):

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    vizwiz = VizWizDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=vizwiz,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader




