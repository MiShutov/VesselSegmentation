import numpy as np
from tqdm import tqdm
import copy


def get_label3d(label_img, label_count, dict_of_equal, x, y, z):
    img3x3x2 = label_img[max(0, x-1):x+2, max(0, y-1):y+2, max(0, z-1):z+2]
    labels = img3x3x2[np.where(img3x3x2>0)].astype(int)
    if len(labels):
        label = min(labels)
        labels = set(labels)
        labels.remove(label)
        
        if labels: 
            dict_of_equal[label] = dict_of_equal[label].union(labels)
        
        ##
        for l in labels:
        #    dict_of_equal[l] = dict_of_equal[l].union([label,])
            label_img[label_img==l] = label
        ##
        return(label)
    else:
        label_count[0]=label_count[0]+1
        dict_of_equal.update({label_count[0]:set()})
        return(label_count[0])

def get_connected(image):
    label_img = np.zeros_like(image)
    label_count = [0,]
    dict_of_equal = {}
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            for z in range(0, image.shape[2]):
                if image[x, y, z]:
                    label_img[x, y, z] = get_label3d(label_img, label_count, dict_of_equal, x, y, z)
    
    return(label_img, dict_of_equal)


def get_equals(label, all_labels, not_checked):
    out = set([label])
    #print(label, all_labels[label], not_checked)
    for l in all_labels[label]:
        out = out.union(get_equals(l, all_labels, not_checked))
    if out:
        return out
    else:
        return [label,]


def get_all_trees(all_labels):
    out = []
    all_labels_copy = copy.deepcopy(all_labels)
    not_checked = list(all_labels_copy.keys())
    while not_checked:
        eq = get_equals(not_checked[0], all_labels_copy, not_checked)
        out.append(eq)
        for x in eq:
            if x in not_checked:
                not_checked.remove(x)

    return(out)  

