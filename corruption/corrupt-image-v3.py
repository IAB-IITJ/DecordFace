# ---------------------------------------------- import necessary libraries

# general
import os
import argparse
import enlighten
from functools import partial

# multiprocessing
import multiprocessing

# data handling
from data_handling.dataset import FRDataset
from torch.utils.data import DataLoader

# image processing
import skimage.io as io
from skimage.util import img_as_ubyte
from skimage.transform import resize

# corruption
from imagenet_c import corrupt
from imagenet_c import corruption_dict

# corruption names
corruption_names = list(corruption_dict.keys())

# we do not consider the weather related corruptions so remove
corruption_names.remove('snow')
corruption_names.remove('frost')
corruption_names.remove('fog')

# ---------------------------------------------- Helper Utils

def corruption(batch, outdir_path, verbose):
  '''
  corrupts the batch of images passed. To be used as `collate_fn` in DataLoader
  :batch: List[Tuple(image, target_path)]
  :outdir_path: what path to append in front of the corrupted image's save target paths
  :verbose: print saving details
  '''

  # corrupting the image in the current batch
  for image, save_target_path in batch:

    # keeping track of original image shape
    ori_image_shape = image.shape[:-1]

    # TODO: FIX
    # resizing image to 224 x 224 since the `corrupt` function expects that size
    image = img_as_ubyte(resize(image, (224, 224), anti_aliasing=True))

    # iterating over severity level
    for sev in range(1,6):

      # iterating over the corruption type
      for corruption_name in corruption_names:
        
        # corrupt image
        corrupt_image = corrupt(
          image, 
          corruption_name=corruption_name, 
          severity=sev
        )

        # creating appropriate save target path
        corr_save_target_path = os.path.join(
          os.path.join(outdir_path, f'{sev}/{corruption_name}'), 
          save_target_path
        )

        # resizing `corrupt_image` to original size since the `corrupt` function expects that size
        corrupt_image = img_as_ubyte(resize(corrupt_image, ori_image_shape, anti_aliasing=True))
        
        # save the corrupted image
        io.imsave(corr_save_target_path, corrupt_image)

        if verbose:
          print('Saved Image at...', corr_save_target_path)

if __name__ == '__main__':

  # ---------------------------------------------- Parsing Command Line Arguments

  # command line argument parser
  parser = argparse.ArgumentParser(description='Corruption Setting')
  parser.add_argument('--indir_path', default='./datasets/data',
    help='''The directory containing images in any folder structure to be corrupted.
    Handles all Face Recognition datasets.''')
  parser.add_argument('--outdir_path', default='./datasets/corrupt-data',
    help='The directory that will contain the corrupted images')
  parser.add_argument('--num_workers', default=multiprocessing.cpu_count(), type=int,
    help='''The number of processes to run in parallel for faster corruption completion,
    one additional process might run if the split is not perfect.
    Tip: Set it to be the number of cores in your processor - i.e. default setting''')
  parser.add_argument('--batch_size', type=int, default=1,
    help='batch size for dataloader of images from the `indir_path`: 1 works the best i.e. default')
  parser.add_argument('--verbose', action='store_true',
    help='weather to print details of what is going on')
  args = parser.parse_args()

  # ---------------------------------------------- Create Corrupted Data Directory Structure

  # create the output data folder hosting all the corrupted data 
  # these folders are common to all `dataset_format`
  print(f'Creating corruption data folders at... {args.outdir_path}')
  print()

  # handles the image loading from `indir_path` and provides a save target path, which needs to be rebased
  # i.e. the {severity}/{corruption type} needs to be appended before the save target path
  corrupt_dataset = FRDataset(args.indir_path, verbose=args.verbose, enable_rebase=True)

  # if `outdir_path` folder doesn't exist then create one
  if not os.path.exists(args.outdir_path):
    os.mkdir(args.outdir_path)

  # creating corruption name as the second level of hierarchy
  for sev in range(1,6):
    for corruption_name in corruption_names:

      # creating severity as the first level of hierarchy
      sev_target_path = os.path.join(args.outdir_path, f'{sev}')

      if not os.path.exists(sev_target_path):
        os.mkdir(sev_target_path)

      # target path for current (severity, corruption name) combo
      sev_corr_target_path = os.path.join(sev_target_path, f'{corruption_name}')

      # creating the dataset for hosting the corrupted images for current (severity, corruption name) combo
      os.mkdir(sev_corr_target_path)

      # create the empty directory structure for current (severity, corruption name) combo
      corrupt_dataset.create_directory_structure(sev_corr_target_path)

  # ---------------------------------------------- Corruption

  # deals with progress bars
  manager = enlighten.get_manager()

  # dataloader for loading the images in batches and corrupts them also within `collate_fn`
  corruption_dataloader = DataLoader(
    corrupt_dataset, 
    collate_fn= partial(
      corruption, 
      outdir_path = args.outdir_path,
      verbose = args.verbose
    ),
    batch_size = args.batch_size,
    num_workers = args.num_workers
  )

  # progress bar for batches
  batch_ticks = manager.counter(total=len(corruption_dataloader), desc="Batches", unit="batch", color="yellow", leave=False)

  for _ in corruption_dataloader:
    
    # update batch progress bar
    batch_ticks.update()

  # done with progress bars
  manager.stop()


