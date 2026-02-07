import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn2
from matplotlib import pyplot as plt

def extract_objects_from_data(path='train.txt'):
    train_files =  [] # 572
    with open(f'ImageSets/{path}', 'r') as fh:
        for line in fh:
            file = line.replace('\n', '')
            train_files.append((file))

    # print(train_files)
    ids = []
    object_class_labels = []
    action_class_labels = []

    for fname in train_files:
        split = fname.split("_")
        ids.append(split[0])
        action_class_labels.append(split[1])
        object_class_labels.append(split[2])

    unique_objects = list(set(object_class_labels))

    unique_object_labels = {}
    for i in range(len(unique_objects)):
        unique_object_labels[i] = unique_objects[i] # index i -> object 45 
    
    return object_class_labels, action_class_labels


## Training
object_class_labels, action_class_labels = extract_objects_from_data()
object_class_labels_test, action_class_labels_test = extract_objects_from_data(path='test.txt')
object_class_labels_val, action_class_labels_val = extract_objects_from_data(path='val.txt')


def plot_histogram(object_class_labels, title):
    counts = Counter(object_class_labels)
    sorted_object_by_freq = sorted(counts.items(), key=lambda x:x[1])
    values = [x[0] for x in sorted_object_by_freq]
    frequencies = [x[1] for x in sorted_object_by_freq]

    plt.bar(values, frequencies)
    plt.xlabel("Objects")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# plot_histogram(object_class_labels, "Training objects")
# plot_histogram(object_class_labels_val, "Validation objects")
# plot_histogram(object_class_labels_test, "Testing objects")

# def comp(list2, list1):
#     for val in list1:
#         if val in list2:
#             print(f"val {val} is also in list 2")

# comp(object_class_labels, object_class_labels_val)

print(len(object_class_labels), len(object_class_labels_val), len(object_class_labels_test))

# import matplotlib.pyplot as plt
# from matplotlib_venn import venn2

def get_venn(A, B, title_A, title_B):

    A = set(A)
    B = set(B)

    subsets_data = (
        len(A - B),   # only in A
        len(B - A),   # only in B
        len(A & B)    # common labels
    )

    venn2(subsets=subsets_data, set_labels=(title_A, title_B), alpha=0.5)
    plt.show()


get_venn(object_class_labels, object_class_labels_test, "Training object classes", "Testing object classes")
get_venn(object_class_labels, object_class_labels_val, "Training object classes", "Validation object classes")


get_venn(action_class_labels, action_class_labels_test, "Training action classes", "Testing action classes")
get_venn(action_class_labels, action_class_labels_val, "Training action classes", "Validation action classes")

