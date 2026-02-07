def get_file_names(filename):
    files =  []
    with open(f'ImageSets/{filename}', 'r') as fh:
        for line in fh:
            file = line.replace('\n', '')
            files.append((file))
    return files

def get_unique_labels(train_files, test_files, task="object_rec"):
    """
    Docstring for get_unique_labels
    
    :param train_files: files names used for training
    :param test_files: files names used for testing
    :param task: object or action recognition

    Returns the total unique labels for both the training and testing classes for the classifier
    Ensure duplicate classes within training and testing are not counted twice
    """
    total_labels = []
    def _get_one_file(files):
        labels = []
        for file in files:
            if task == "object_rec":
                label = file.split("_")[2] # object
            elif task == "action_rec":
                label = file.split("_")[1] # action
            labels.append(label)
        return labels
    
    total_labels = _get_one_file(train_files) + _get_one_file(test_files)
    return list(set(total_labels))