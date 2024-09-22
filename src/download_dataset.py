import os
import tarfile
import urllib.request
import shutil

def download_dataset(url, raw_folder):
    # create raw folder if it does not exists
    os.makedirs(raw_folder, exist_ok=True)

    # download the dataset
    print("Downloading LFW dataset...")
    filename = url.split('/')[-1]
    filepath = os.path.join(raw_folder, filename)
    urllib.request.urlretrieve(url, filepath)

    #Extarct the dataset
    print("Extracting LFW dataset...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=raw_folder)

    #remove the downloaded tar file
    print("Removing the downloaded tar file...")
    os.remove(filepath)

    # rename the extracted folder to 'lfw'
    extracted_folder = os.path.join(raw_folder, 'lfw-deepfunneled')
    lfw_folder = os.path.join(raw_folder, 'lfw')
    # os.rename(extracted_folder, lfw_folder)
    shutil.move(extracted_folder, lfw_folder)

    print("LFW dataset downloaded and extracted successfully to {}".format(lfw_folder))

def organize_dataset(raw_folder, processed_folder):
    lfw_folder = os.path.join(raw_folder, 'lfw')

    # create processed folder if it does not exists
    os.makedirs(processed_folder, exist_ok=True)

    # copy and organize the dataset
    print("Organizing LFW dataset...")

    for person_name in os.listdir(lfw_folder):
        src_folder = os.path.join(lfw_folder, person_name)
        dst_folder = os.path.join(processed_folder, person_name)

        if os.path.isdir(src_folder):
            shutil.copytree(src_folder, dst_folder)

    print("LFW dataset organized successfully to {}".format(processed_folder))


if __name__ == "__main__":
    #setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_folder = os.path.join(project_root, 'data', 'raw')
    processed_folder = os.path.join(project_root, 'data', 'processed', 'aligned_faces')

    #lfw dataset url (deepfunneled version)
    url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'

    # download and extract the dataset
    download_dataset(url, raw_folder)

    # organize the dataset
    organize_dataset(raw_folder, processed_folder)

