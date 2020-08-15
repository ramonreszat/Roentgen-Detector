import os, glob, re
import numpy as np

import pandas as pd
import pydicom as dicom

from mxnet import nd
from mxnet import gluon
from mxnet.image import CreateAugmenter

class DICOMFolderDataset(gluon.data.dataset.Dataset):
  def __init__(self, data_dir, labels, flag=1):
    self._instances = []
    self._labels = pd.read_csv(labels, index_col='sop_instance_uid')

    # pick instances with matching SOPInstanceUID
    for instance in glob.glob(data_dir, recursive=True):
      if os.path.splitext(os.path.split(instance)[1])[0] in self._labels.index and instance not in self._instances:
        self._instances.append(instance)

  def __getitem__(self, idx):
    study = dicom.dcmread(self._instances[idx], defer_size="512 MB")
    diagnosis = self._labels.loc[study.SOPInstanceUID]['diagnosis']
    bbox = self._labels.loc[study.SOPInstanceUID]['bbox']

		# convert image from uint8 to float32
    img = study.pixel_array.astype('float32')/255

    # positive datapoints have a tuple (x,y,w,h) as prediction target
    if diagnosis == 'P':
      bbox = tuple(map(int, re.findall(r'[0-9]+', bbox)))
      # float32 ratios for the bounding box
      bbox = (bbox[0]/1024,bbox[1]/1024,bbox[2]/1024,bbox[3]/1024)
    else:
      bbox = tuple((0.0,0.0,0.0,0.0))

    return nd.array(img, dtype=np.float32), nd.array(bbox, dtype=np.float32)

  def __len__(self):
    return len(self._instances)