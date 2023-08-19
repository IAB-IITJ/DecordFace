# ---------------------------------------------- import necessary libraries

# general
import os
from tqdm import tqdm

# matrix manipulation
import numpy as np

# data handling
from torch.utils.data import Dataset

# image processing
import skimage.io as io

# ---------------------------------------------- Dataset Handling

# image extensions supported
img_extensions = ('bmp', 'jpe', 'jp2', 'tiff', 'tif', 'sr', 'ras', 'pbm', 'pgm', 'ppm', 'png', 'jpeg', 'jpg')

# the dataset class
class FRDataset(Dataset):
    """
    Handles the popular FR Datasets.
    """
    def __init__(self, indir_path, outdir_path=None, verbose=False, enable_rebase=False):
        """
        :indir_path: the path to the input data folder, can be in any format. Make sure your path don't end with `/`
        :outdir_path: the path to the output data folder (can be `None` only if `enable_rebase` is set `True`). Make sure your path don't end with `/`
        :verbose: if True, prints all the details in each function call 
        :enable_rebase: if True, doesn't appends `outdir_path` to the save target paths to allow 
            appending own path at the starting of the save target paths
        """
        self.indir_path = indir_path
        
        if not enable_rebase and outdir_path is None:
            raise Exception('`outdir_path` cannot be None with `enable_rebase = False`')
        
        if enable_rebase and outdir_path is not None:
            raise Exception('`outdir_path` must be None with `enable_rebase = True`')
        
        self.outdir_path = outdir_path
        self.verbose = verbose
        self.enable_rebase = enable_rebase

        # think of this step as indexing the `indir_path`
        if self.verbose:
            print(f'Indexing... {self.indir_path}')

        # list of all image paths in the `indir_path`: retrive location
        self.image_paths = list()

        # list of all image paths in the `outdir_path`: target location for saving
        self.save_image_paths = list()

         # walk through the `indir_path`
        for dirpath, _, filenames in tqdm(os.walk(self.indir_path)):

            # iterate through all the image files in the current directory
            for filename in filenames:
                
                # if the file is an image
                if filename.endswith(img_extensions):

                    # append the image path to the list of all image paths
                    self.image_paths.append(os.path.join(dirpath, filename))
                    
                    # create the save target path
                    save_target_path = os.path.join(dirpath[(len(self.indir_path)+1):], filename)
                    
                    # if rebase is enabled then append as it is otherwise add the `outdir_path` at the starting
                    if self.enable_rebase:
                        self.save_image_paths.append(save_target_path)
                    else:
                        self.save_image_paths.append(os.path.join(self.outdir_path, save_target_path))


        if self.verbose:
            print('Indexed!')
            print(f'Number of image files found in {indir_path}: {len(self.image_paths)}')

    def create_directory_structure(self, outdir_path=None):
        """
        creates the same directory structure as `indir_path` at the `outdir_path`
        :outdir_path: (deafult: None, uses `self.outdir_path`) 
            if passed indicates where the directory structure should be created
        """ 
        inputpath = self.indir_path
        outputpath = self.outdir_path

        if self.enable_rebase and outdir_path is None:
            raise Exception('`outdir_path` cannot be None with `self.enable_rebase = True`')
        
        if not self.enable_rebase and outdir_path is not None:
            raise Exception('''
            ignoring `outdir_path` in this function call, using `self.outdir_path`.
            In order to use `outdir_path` in this function call set `self.enable_rebase = False`
            ''')
        
        if self.enable_rebase:
            outputpath = outdir_path
        
        if self.verbose:
            print(f'Creating same directory structure as {inputpath} at... {outputpath}')

        # create the empty directory structure same as that in `inputpath`
        for dirpath, _, _ in os.walk(inputpath):
            structure = os.path.join(outputpath, dirpath[(len(inputpath)+1):])
            if not os.path.isdir(structure):
                os.mkdir(structure)
            else:
                print("Folder does already exits!")
        
        if self.verbose:
            print('Created!')

    def __getitem__(self, idx):
        """
        returns an image (every image is converted to 3 color channels) from the dataset
        """
        # read the image in RGB format
        image = io.imread(self.image_paths[idx])

        # dealing with image which has the color channel missing
        if len(image.shape) == 2:
            image = np.repeat(image[:,:,np.newaxis], 3, axis=-1)
        
        # dealing with single color channel image
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        if self.verbose:
            print(f'Retrived Image from... {self.image_paths[idx]}')
        
        # returns the image and target image path of that image
        return image, self.save_image_paths[idx]
    
    def __len__(self):
        """
        returns the number of image files in the `indir_path` folder
        """
        return len(self.image_paths)