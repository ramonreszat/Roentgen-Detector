import os
import boto3
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--aws_id', '-id', type=str)
parser.add_argument('--access_key', '-key', type=str)

import zipfile
import shutil

args = parser.parse_args()

# place the login information for accessing the S3 bucket
s3r = boto3.resource('s3', aws_access_key_id=args.aws_id, aws_secret_access_key=args.access_key)
bucket = s3r.Bucket('siim-acr-pneumothorax')

os.mkdir('siim-acr-data')

# fetch the development set
bucket.download_file('dev-sample.csv', 'siim-acr-data/dev-sample.csv')
bucket.download_file('train-data/pneumothorax-devset.zip', 'siim-acr-data/pneumothorax-devset.zip')

# fetch the training data
bucket.download_file('train-sample.csv', 'siim-acr-data/train-sample.csv')
bucket.download_file('train-data/pneumothorax-trainset.zip', 'siim-acr-data/pneumothorax-trainset.zip')

# see DICOM folder datset for an overview of the unzipped directory structure
zipfile.ZipFile('siim-acr-data/pneumothorax-devset.zip', 'r').extractall('siim-acr-data/dev-pneumothorax')
zipfile.ZipFile('siim-acr-data/pneumothorax-trainset.zip', 'r').extractall('siim-acr-data/train-pneumothorax')
os.remove('siim-acr-data/pneumothorax-devset.zip')
os.remove('siim-acr-data/pneumothorax-trainset.zip')

# delete hidden archive folder
shutil.rmtree('siim-acr-data/dev-pneumothorax/__MACOSX')
shutil.rmtree('siim-acr-data/train-pneumothorax/__MACOSX')