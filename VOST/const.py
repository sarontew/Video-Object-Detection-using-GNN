train_files =  [] # 572
with open('ImageSets/train.txt', 'r') as fh:
    for line in fh:
        file = line.replace('\n', '')
        train_files.append((file))

# Maps video names to class ids
class_labels = {
    f : train_files.index(f)
    for f in train_files
}
