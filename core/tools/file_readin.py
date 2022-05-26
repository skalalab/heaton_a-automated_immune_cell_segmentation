import os
import re
import codecs
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def regex_dictionary_builder(keys = [], values = []):
    """
    Regex dictionary builder: builds a dictionary of human searchable terms corresponding to regex searchable queries.
    Length of keys and values list must be the same, and related terms must match positionally for the method to work.

    Parameters
    ----------
    keys: list
        List of human searchable terms, must be same length as values list
    values: list
        List of regex searchable queries, must be same length as keys list

    Returns
    -------
    channels_dict: dictionary
        Dictionary of human searchable keys corresponding to regex searchable queries as values.

    """

    if len(keys) != len(values):
        # Change to be a zip of the 2 lists
        raise ValueError('Length of keys and value lists must be the same!')
    channels_dict = {keys[i]:values[i] for i in range(len(keys))}
    return channels_dict


def channel_tester(path, channels, raw_data_directory = 'raw_data', display_paths = False):
    """
    Channel tester: Simple method to confirm that channels (and more specifically channel regex patterns) are being structured correctly.

    Simply iterate through as one would in the file_readin method, and just print channel vs. file that would have been read in from the directory for sanity check.
    Test Regex formulas and patterns here: https://regex101.com by copying in the expression for a given channel and the line from the display files.

    Examples
    --------
        Pattern: [SingleExp]*mCherry.*photons
        Test string:
            channel_keys = ['mCherry_t2', 'mCherry_photons', 'NADH', 'FAD', 'mCherry_a1_single_exp', 'mCherry_photons_single_exp', 'mCherry_t1_single_exp']
            channel_regex_values = [r'[SingleExp]*mCherry.*t2', r'[SingleExp]*mCherry.*photons', r'[SingleExp]*NADH.*asc', r'[SingleExp]*FAD.*asc', r'SingleExp.*a1',r'SingleExp.*photons', r'SingleExp.*t1']
            channels = {channel_keys[i]:channel_regex_values[i] for i in range(len(channel_keys))}

    Parameters
    ----------
    path: string
        Path to test read in from.
    channels: dictionary
        Dictionary of channel search terms and regex patterns to be read in.
    raw_data_directory: string, optionoal
        Specific folder for raw data to be read in from; must contain all relevant raw data to be read in.
        Default is 'raw_data'
    display_paths: bool, optional
        Bool to display full paths for quick validation.
        Default is False.

    Returns
    -------
    None

    """

    # curr_dir_list and dir_counter help keep track of where we are in a given directory:
    #   - they will be updated each time we have a new set of directories, and then saved along with the current fov/slice dictionary
    curr_dir_list = list()
    dir_counter = 0
    for root, dirs, files in os.walk(path):
        curr_file_name = ''
        if display_paths: print(files)
        if raw_data_directory in root:
            # Update if we have a new set of child directories
            if dirs:
                curr_dir_list = dirs
                dir_counter = 0
                if display_paths:
                    print(curr_dir_list)
                    print('_________________________________________________________________________________________________________________________________________________________')
             # Once we hit files, this is effectively a 'leaf' in this implementation, no additional subdirectories so normally begin read in
             # (here just generate paths)
            if files:
                # Use the same structure as file_readin for generating the data paths
                temp_composite = [f for f in files if(re.search(r'composite.tif', f, re.IGNORECASE))][0]
                print(temp_composite + ' : Composite.tif')
                temp_ground_truth = [f for f in files if(re.search(r'.*ground_truth.tiff', f, re.IGNORECASE))][0]
                print(temp_ground_truth + ' : ground_truth.tif')
                for channel in channels:
                    print(channel)
                    print([f for f in files if(re.search(channels.get(channel), f))])
                    temp_path = [f for f in files if(re.search(channels.get(channel), f))][0]
                    print(temp_path + ' : ' + channel)
                curr_file_name = curr_dir_list[dir_counter]
                curr_root_dir = os.path.abspath(root)
                dir_counter+=1
                print('----------------------------------------------------------------------------------------------------------------')


def file_readin_old(path, channels, raw_data_directory = 'raw_data', display_paths = False, display_images = False):
    """
    File readin: Recursively reads in all files into a single dictionary for each experiment

    For a given subdirectory within the path specified, automatically perform the following:
        - Build an (n+2)-length dictionary consisting of files in the order given from the channels list (n channels + ground truth + composite)

    Parameters
    ----------
    path: string
        Path to read in files from.
    channels: dictionary
        Dictionary of channel search terms and regex patterns to be read in.
    raw_data_directory: string, optionoal
        Specific folder for raw data to be read in from; must contain all relevant raw data to be read in.
        Default is 'raw_data'.
    display_paths: bool, optional
        Bool to display full paths for quick validation.
        Default is False.
    display_images: bool, optional
        Bool to display inline images for quick validation.
        Default is False.

    Returns
    -------
    im_list: list
        List of dictionaries containing numpy arrays for composite, ground truth, and n additional channel images associated with original channel keys passed in.
    """

    # Final list that we'll build over time
    im_list = list()
    # curr_dir_list and dir_counter help keep track of where we are in a given directory:
    #   - they will be updated each time we have a new set of directories, and then saved along with the current fov/slice dictionary
    curr_dir_list = list()
    dir_counter = 0
    for root, dirs, files in os.walk(path):
        curr_file_name = ''
        if display_paths: print(files)
        if raw_data_directory in root:
            # Update if we have a new set of child directories
            if dirs:
                curr_dir_list = dirs
                dir_counter = 0
                if display_paths:
                    print(curr_dir_list)
                    print('_________________________________________________________________________________________________________________________________________________________')
            # Once we hit files, this is effectively a 'leaf' in this implementation, no additional subdirectories so begin read in
            if files:
                # Dictionary that will contain all data for this slice/FOV
                curr_fov_stack = dict()
                # Read composite/ground truth by building the path (if the regex is non-null)
                temp_composite = np.array(Image.open(os.path.abspath(os.path.join(root,[f for f in files if(re.search(r'composite', f, re.IGNORECASE))][0]))))
                print('found composite')
                temp_ground_truth = np.array(Image.open(os.path.abspath(os.path.join(root,[f for f in files if(re.search(r'.*ground_truth.tiff', f, re.IGNORECASE))][0]))))
                print('found ground truth')
                # .asc channels can be read in as follows
                for channel in channels.keys():
                    # Check for T1 edge case
                    print(channel)
                    if channel == 'mCherry_t1':
                        # Prefer codecs method here to avoid readin issues for T1 image
                        with codecs.open(os.path.abspath(os.path.join(root,[f for f in files if(re.search(channels.get(channel), f))][0])), encoding='utf-8-sig') as f:
                            curr_image = np.array([[float(x) for x in line.split()] for line in f])
                        # Filter to specific mCherry lifetime
                            curr_fov_stack.update({channel: curr_image})
                    # Otherwise proceed as normal with numpy genfromtext and same path building mechanism
                    else:
                        curr_fov_stack.update({channel: np.genfromtxt(os.path.abspath(os.path.join(root, [f for f in files if(re.search(channels.get(channel), f))][0])), dtype='i8')})
                curr_file_name = curr_dir_list[dir_counter]
                curr_root_dir = os.path.abspath(root)
                # After reading all images into the list, condense into a single 3D array and then create a dictionary of all relevant file info
                curr_fov_stack.update({'composite':temp_composite, 'ground_truth':temp_ground_truth, 'root_dir':curr_root_dir, 'file_name':curr_file_name})
                im_list.append(curr_fov_stack)
                if display_images:
                    plt.figure(figsize=(20,20))
                    plt.title(curr_file_name)
                    plt.imshow(curr_fov_stack.get('composite'))
                    plt.figure(figsize=(20,20))
                    plt.title(curr_file_name + ' (Ground truth)')
                    plt.imshow(curr_fov_stack.get('ground_truth'), cmap=plt.cm.Greys_r)
                    fig, axs = plt.subplots(2, 2)
                    fig.set_figheight(20)
                    fig.set_figwidth(20)
                    axs[0, 0].imshow(curr_fov_stack.get('mCherry_t1_single_exp'), cmap=plt.cm.Greys_r)
                    axs[0, 0].set_title('mCherry T1')
                    axs[0, 1].imshow(curr_fov_stack.get('mCherry_photons'), cmap=plt.cm.Greys_r)
                    axs[0, 1].set_title('mCherry Photons')
                    axs[1, 0].imshow(curr_fov_stack.get('NADH'), cmap=plt.cm.Greys_r)
                    axs[1, 0].set_title('NADH Photons')
                    axs[1, 1].imshow(curr_fov_stack.get('FAD'), cmap=plt.cm.Greys_r)
                    axs[1, 1].set_title('FAD Photons')
                    plt.show()
                dir_counter+=1
        else:
            pass
    return (im_list)


def file_readin(gt_list,
                photon_list,
                mCherry_t1_list, 
                composite_list):
    accumulator_dict = dict()
    str_gt_list = [str(x) for x in gt_list]
    str_t1_list = [str(x) for x in mCherry_t1_list]
    str_comp_list = [str(x) for x in composite_list]

    for mC_pi in photon_list: 
        curr_dict = dict()
        curr_stem = mC_pi.parents[0].stem
        curr_dict['photons'] = np.genfromtxt(mC_pi)
        for gt_path in str_gt_list: 
            re_result = re.search(rf'\b{curr_stem}\b', gt_path)
            if re_result:
                curr_dict['gt'] = tifffile.imread(gt_path)
                print(f'Read in: {gt_path}')
        for t1_path in str_t1_list: 
            re_result = re.search(rf'\b{curr_stem}\b', t1_path)
            if re_result:
                curr_dict['t1'] = np.genfromtxt(t1_path)
                print(f'Read in: {t1_path}')
        for comp_path in str_comp_list: 
            re_result = re.search(rf'\b{curr_stem}\b', comp_path)
            if re_result:
                curr_dict['composite'] = tifffile.imread(comp_path)
                print(f'Read in: {comp_path}')
        curr_dict['file_name'] = curr_stem
        accumulator_dict[curr_stem] = curr_dict
    return accumulator_dict