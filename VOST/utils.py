def get_file_names(filename):
    files =  []
    with open(f'ImageSets/{filename}', 'r') as fh:
        for line in fh:
            file = line.replace('\n', '')
            files.append((file))
    return files