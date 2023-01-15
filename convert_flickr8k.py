from magma.datasets.convert_datasets import convert_dataset
import csv
from pathlib import Path

def my_dataset_iterator():
    """
    Implement an iterator for your dataset that for every datapoint yields a tuple
    image_path, {"captions": [...], "metadata": {...}, }, where image_path is the path to the image as a Path object, captions is a list of caption strings and metadata is an optional field.
    """
    with open("/gpfs/alpine/csc499/proj-shared/magma/flickr8k/captions.txt") as f:
        default_iter = csv.reader(f)

        custom_iter = []
        next(default_iter)
        for row in default_iter:
             custom_iter.append((Path('/gpfs/alpine/csc499/proj-shared/magma/flickr8k/images/' + row[0]), {"captions": row[1]}))
        return iter(custom_iter)


if __name__ == "__main__":
    convert_dataset(data_dir="/gpfs/alpine/csc499/proj-shared/magma/flickr8k_processed", ds_iterator=my_dataset_iterator(), mode='cp')
