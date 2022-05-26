import os 
import cv2 
import time 
import pprint
import tifffile 
import operator 
import itertools
import matplotlib 

import numpy as np 
import scipy.ndimage as ndi 
import ipywidgets as widgets 
import matplotlib.pyplot as plt

from PIL import Image
from scipy import misc 
from ipywidgets import Box
from skimage.filters import * 
from core.tools.metrics import * 
from skimage.feature import *
from skimage.morphology import * 
from skimage.segmentation import *
from matplotlib.patches import Patch
from skimage.measure import label, regionprops 


# Create list of all metrics
ALL_METRICS_LIST = ['Dice', 
                   'Jaccard']


# Create dictionary of all thresholding functions, pull whatever is needed
ALL_THRESHOLDING_METHODS = {'min' : threshold_minimum, 
                            'yen' : threshold_yen,
                            'otsu' : threshold_otsu, 
                            'local_otsu' : rank.otsu,
                            'local' : threshold_local,
                            'triangle' : threshold_triangle,
                            'multi_otsu' : threshold_multiotsu}


# Define all local thresholding methods: 
ALL_LOCAL_THRESHOLDING_METHODS = ['mean', 
                                  'median',
                                  'generic', 
                                  'gaussian']


def post_thresholding_processing_helper(thresh_curr, 
                                        min_roi_size=100, 
                                        filter_small_items=True, 
                                        local_otsu=False, 
                                        method_key='local',
                                        composite_im=None,
                                        save_intermediates=False,
                                        save_intermediates_fname=None):
    '''
    Post thresholding processing: applies a series of steps as follows following thresholding...
        - Binary closing to ensure good filling of regions of interest (ROIs)
        - Border clearing to remove noise
        - Edge detection through a canny operator
        - Binary filling of holes and cell profiler style ROI labeling
        - Small item filtering to remove areas of the image that are likely extra noise picked up

    Parameters
    ----------
    mCherry_photons: ndarray
        Reference mCherry photons image.
    thresh_curr: ndarray
        Current thresholded image array to work on.
    min_roi_size: int, optional
        Minimum size for an ROI; filter out ROIs smaller than this to remove noise.
        Default is 100 pixels.

    Returns
    -------
    filtered_labeled_curr: ndarray
        Numpy ndarray containing a cellprofiler style labeled image wtih small items filtered out.
    '''

    # Perform binary closing on the thresholded image (with padding)
    binary_closing_curr = binary_closing(thresh_curr, square(3))
    # Clear border elements (i.e. anything on the border that is either a half cell, etc.) will be removed
    # This may not be as effective on the padded image...
    cleared_border_curr = clear_border(binary_closing_curr)

    edges = canny(image=cleared_border_curr,
                                  sigma=2,
                                  low_threshold=0.01,
                                  high_threshold=0.1)
    binary_filled_edges = ndi.binary_fill_holes(ndi.binary_dilation(edges))

    if method_key == 'multi_otsu':
        if save_intermediates: 
            plt.imsave(f'output_files\\intermediate_images\\{save_intermediates_fname}_canny_edges_image.png', edges[10:-10, 10:-10], cmap=plt.cm.Greys_r)
            plt.imsave(f'output_files\\intermediate_images\\{save_intermediates_fname}_binary_closing_labeled_image.png', ndi.label(binary_filled_edges)[0][10:-10, 10:-10], cmap=plt.cm.Greys_r)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_filled_edges.astype(np.uint8),
                                                                               connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    if filter_small_items:
        # Accumulator array of sorts that we build as we iterate through the connected components
        filtered_curr = np.zeros(output.shape, dtype=np.uint8)
        # Filter out small items based on min_roi_size
        for i in range(0, nb_components):
            if sizes[i] >= min_roi_size:
                filtered_curr[output == i + 1] = 255
            if local_otsu:
                if sizes[i] >= 1600:
                    filtered_curr[output == i + 1] = 0
        # Remove padding and label image for individual ROIs
        if method_key == 'multi_otsu':
            if save_intermediates:
                plt.imsave(f'output_files\\intermediate_images\\{save_intermediates_fname}_small_item_filtered_image.png', clear_border(ndi.label(filtered_curr, output=np.uint8)[0][10:-10, 10:-10]), cmap=plt.cm.Greys_r)
                plt.imsave(f'output_files\\intermediate_images\\{save_intermediates_fname}_adaptive_segmentation_overlay_image.png', mark_boundaries(composite_im, clear_border(ndi.label(filtered_curr, output=np.uint8)[0][10:-10, 10:-10]), color=(1,1,0), mode="thick"))
        return clear_border(filtered_curr[10:-10, 10:-10]), clear_border(ndi.label(filtered_curr, output=np.uint8)[0][10:-10, 10:-10])
    else:
        return clear_border(output[10:-10, 10:-10]), clear_border(ndi.label(output)[0][10:-10, 10:-10])


def post_thresholding_processing_driver(curr_dict, 
                                        filter_small_items=True, 
                                        min_roi_size=100, 
                                        save_intermediates=False,
                                        composite=None, 
                                        fname=None):
    results_accumulator_dict = dict()
    for curr_dict_key in curr_dict.keys():
        if curr_dict_key == 'local_otsu':
            local_otsu = True
        else: 
            local_otsu = False
        im, results_accumulator_dict[curr_dict_key] = post_thresholding_processing_helper(curr_dict.get(curr_dict_key),
                                                                                        filter_small_items=filter_small_items,
                                                                                        min_roi_size=min_roi_size,
                                                                                        local_otsu=local_otsu,
                                                                                        method_key=curr_dict_key,
                                                                                        composite_im=composite,
                                                                                        save_intermediates=save_intermediates,
                                                                                        save_intermediates_fname=fname)
    return im, results_accumulator_dict


def combination_testing_helper(curr_dict, size=(512, 512)):
    curr_combo_accumulator_dict = dict()
    for curr_dict_counter in range(1, len(curr_dict.keys()) + 1):

        curr_combo_list = [list(x) for x in itertools.combinations(curr_dict.keys(), curr_dict_counter)]
        for combos in curr_combo_list:
            max_intensity_proj_array = np.zeros(size, dtype=np.uint8)
            for combo in combos:
                max_intensity_proj_array = np.max(np.stack([max_intensity_proj_array,
                                                            curr_dict.get(combo)],
                                                           axis=2), axis=2)
            curr_combo_accumulator_dict[tuple(combos)] = max_intensity_proj_array
    return curr_combo_accumulator_dict


def optimize_local_thresholding(image_list,
                                block_size_start=1,
                                block_size_end=102, 
                                block_size_list=None,
                                ground_truth_key='gt',
                                photon_image_key='photons'):

    # If our block size range is either less than 2, or if either the
    # block size start or end are less than feasible values, auto
    # set to default values; must be positive odd integer values
    # see: https://scikit-image.org/docs/0.13.x/api/skimage.filters.html#skimage.filters.threshold_local
    if (block_size_end-block_size_start <= 2
        or block_size_start < 1 
        or block_size_end < 3 
        or block_size_list is None): 

        block_size_start = 1
        block_size_end = 102
        temp_block_size_list = np.arange(block_size_start, 
                                         block_size_end)
        block_size_list = temp_block_size_list[temp_block_size_list % 2 == 1] 

    # Create final list of optimal block sizes for each image
    final_block_list = list() 

    # For each image dictionary in the total image list, iterate through 
    # and calculate a sample mask through the quickest set of steps; 
    # store optimal block size for every image. 
    for ix, image_dict in enumerate(image_list): 
        
        best_dice = 0
        best_block_size = 0

        # Pull and store current image and ground truth once
        curr_im_local = image_dict.get(photon_image_key)  
        curr_gt_local = image_dict.get(ground_truth_key)
        # Check every block size in our block size list
        for curr_block_size in block_size_list: 
            print(curr_block_size)
            # Check every local thresholding method requested
            for curr_method in ALL_LOCAL_THRESHOLDING_METHODS: 

                # Repeat steps as normal with default parameters
                curr_im_padded_local = pad_array(curr_im_local)
                curr_im_thresh_local = threshold_local(curr_im_padded_local, 
                                                 curr_block_size,
                                                 method=curr_method)

                # post thresholding processing driver

                # combination testing helper 

                temp_dice = calculate_dice(curr_im_thresh_local[10:-10, 10:-10], curr_gt_local)

                if best_dice < temp_dice: 
                    best_dice = temp_dice
                    best_block_size = curr_block_size 
                print(f'image number: {ix}; block size {curr_block_size}; number: {curr_method}')
            print(best_dice)
            print('___________________________________________________________')
        final_block_list.append(best_block_size)
    
    return final_block_list


def adaptive_segmentation(image_list, 
                          save_masks=False, 
                          metric_list=None,
                          min_roi_size=100,
                          cellpose_data=None, 
                          local_otsu_radius=25,
                          multi_otsu_classes=2,
                          include_lifetime=False, 
                          include_cellpose=False, 
                          threshold_methods='all',
                          pre_threshld_image=True,
                          export_metrics_csv=True,
                          lifetime_start_value=1300,
                          lifetime_end_value=1500,
                          save_comparison_images=False, 
                          optimize_pre_threshold=False,
                          display_comparison_images=False,
                          export_intermediate_images=False,
                          export_intermediate_indices=None,
                          local_threshold_block_size_start=1,
                          local_threshold_block_size_end=102,
                          local_threshold_block_methods='all',
                          local_threshold_block_size_list=None,
                          **additional_datasets): 
    
    # Set up final locations for data to be stored
    file_name_list = list() 
    final_data_list = list() 
    best_method_list = list() 
    intermediate_data_list = list() 

    # Create timer 
    total_time = 0
    if export_intermediate_images: 
        for intermediate_ix, fname in export_intermediate_indices.items(): 
            print(intermediate_ix, fname)
            plt.imsave(f'output_files\\intermediate_images\\{fname}_photon_image.png', image_list[intermediate_ix].get('photons').astype(int), cmap=plt.cm.Greys_r)

    # WARNING: THIS TAKES MULTIPLE HOURS TO COMPLETE, 
    # ONLY SET IF ABSOLUTELY SURE YOU NEED IT.
    if optimize_pre_threshold: 

        local_threshold_block_size_list = optimize_local_thresholding(image_list, 
                                        local_threshold_block_size_start,
                                        local_threshold_block_size_end,
                                        local_threshold_block_methods)

    elif local_threshold_block_size_list is None: 
        # Set to arbitrary value of 19
        local_threshold_block_size_list = [19]*len(image_list)

    # Set up metrics list if None are defined: 
    if metric_list is None: 

        metric_list = ALL_METRICS_LIST
    
    # Create dictionary of all thresholding methods to test (all/subset of all)
    curr_threshold_methods = dict() 

    # If we want all thresholding methods, then set to constant 
    # Also check for None/empty list
    # TODO: throw error here if empty or None, strict type check for list 
    if (threshold_methods == 'all' 
        or threshold_methods is None 
        or threshold_methods == []):

        curr_threshold_methods = ALL_THRESHOLDING_METHODS
    
    # Otherwise check for list and iterate through pulling 
    elif isinstance(threshold_methods, list) and len(threshold_methods) > 0: 
        for method_key in threshold_methods: 
            curr_threshold_methods[method_key] = ALL_THRESHOLDING_METHODS.get(method_key)

    # Loop over every image in the image_list
    for i, image_dict in enumerate(image_list): 
        
        start_time = time.time() 
        curr_intermediate_dict = dict()
        
        # Append whatever our current file is to the file_name_list 
        file_name_list.append(image_dict.get('file_name'))

        # Pull relevent images for procesing: 
        ground_truth = image_dict.get('gt').astype(int)
        mCherry_photons = image_dict.get('photons').astype(int)
        mCherry_lifetime_t1 = image_dict.get('t1')
    
        # Pad mCherry photons image with 10 pixel wide  
        # buffer of 0's to reduce effects of border cells 
        if pre_threshld_image: 
            # Apply a simple local threshold to the image prior to padding
            temp_local_threshold = threshold_local(mCherry_photons, 
                                                   local_threshold_block_size_list[i])
            mCherry_photons_padded = pad_array(temp_local_threshold)

        else: 
            # Otherwise, just pad as normal 
            mCherry_photons_padded = pad_array(mCherry_photons)

        # Establish relevant thresholding parameters 
        # before entering for loop for methods: 
        selem = disk(local_otsu_radius)

        threshold_intermediate_dict = dict() 

        for method_key, method in curr_threshold_methods.items():
            if method_key == 'multi_otsu': 
                # TODO: potentially add multiple class setup for multiotsu here? 
                curr_im_thresh_temp = method(mCherry_photons_padded, 
                                             classes=multi_otsu_classes)
                curr_im_thresh = np.digitize(mCherry_photons_padded,
                                             bins=curr_im_thresh_temp)
            elif method_key == 'local_otsu': 
                curr_im_thresh_temp = method(mCherry_photons_padded, selem)
            elif method_key == 'local':
                curr_im_thresh = method(mCherry_photons_padded, local_threshold_block_size_list[i])
            else: 
                curr_im_thresh_temp = method(mCherry_photons_padded)
            
            curr_im_thresh = mCherry_photons_padded >= curr_im_thresh_temp

            threshold_intermediate_dict[method_key] = curr_im_thresh
        
        if include_lifetime: 
            # Copy array and filter to specific lifetime values and pad
            mCherry_lifetime_t1_filtered = mCherry_lifetime_t1.copy()
            mCherry_lifetime_t1_filtered[mCherry_lifetime_t1_filtered < lifetime_start_value] = 0
            mCherry_lifetime_t1_filtered[mCherry_lifetime_t1_filtered > lifetime_end_value] = 0

            mCherry_lifetime_t1_padded = pad_array(mCherry_lifetime_t1_filtered)
            if export_intermediate_images: 
                if i in list(export_intermediate_indices.keys()): 
                    plt.imsave(f'output_files\\intermediate_images\\{export_intermediate_indices.get(i)}_t1_image.png', mCherry_lifetime_t1_padded[10:-10, 10:-10], cmap=plt.cm.Greys_r)

            for curr_key, curr_thresh_im in threshold_intermediate_dict.items(): 
                threshold_intermediate_dict[curr_key] = np.logical_and(mCherry_lifetime_t1_padded, curr_thresh_im)

        curr_intermediate_dict['thresholded_images'] = threshold_intermediate_dict
        
        if export_intermediate_images: 
            if i in list(export_intermediate_indices.keys()): 
                export = True
                fname = export_intermediate_indices.get(i)
                plt.imsave(f'output_files\\intermediate_images\\{export_intermediate_indices.get(i)}_fg_bg_sep_image.png', threshold_intermediate_dict.get('multi_otsu')[10:-10, 10:-10], cmap=plt.cm.Greys_r)

            else: 
                export = False
                fname = None
        else: 
            pass
        from matplotlib import colors

        post_thresh_proc_dict = post_thresholding_processing_driver(threshold_intermediate_dict,
                                                                    filter_small_items=True,
                                                                    min_roi_size=min_roi_size, 
                                                                    save_intermediates=export, 
                                                                    composite=image_dict.get('composite'),
                                                                    fname=fname)
        results_accumulator_dict = combination_testing_helper(post_thresh_proc_dict[1],
                                                              post_thresh_proc_dict[1].get(
                                                                  next(iter(post_thresh_proc_dict[1]))).shape)

        total_time += time.time() - start_time

        #test_intersection = np.pad(np.logical_and((mCherry_photons_padded > threshold_otsu(mCherry_photons_padded)).astype(bool)[10:-10, 10:-10], best_method.astype(bool)), 10, mode='constant', constant_values=(0, 0))

        curr_stats_dict, best_method_index, best_method = calculate_image_statistics_helper(ground_truth, 
                                                                                            results_accumulator_dict,
                                                                                            *metric_list)
        fig = plt.figure(figsize=(512,512), dpi=1, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(ground_truth, cmap=plt.cm.Greys_r)
        multi_otsu_binary = post_thresh_proc_dict[1].get('multi_otsu').copy()
        multi_otsu_binary[multi_otsu_binary > 0] = 1
        cmap = colors.ListedColormap(['black', 'blue'])
        ax.imshow(multi_otsu_binary, cmap, interpolation='none', alpha=0.3)
        yen_binary = post_thresh_proc_dict[1].get('yen').copy()      
        yen_binary[yen_binary > 0] = 100
        cmap = colors.ListedColormap(['black', 'red'])
        ax.imshow(yen_binary, cmap, interpolation='none', alpha=0.3)
        final_data_list.append(curr_stats_dict)
        if export_intermediate_images: 
            if i in list(export_intermediate_indices.keys()): 
                fig.savefig(f'output_files\\intermediate_images\\{export_intermediate_indices.get(i)}_ensemble_demo_image.png', dpi=1)


        if not display_comparison_images and not save_masks and not save_comparison_images:
            pass
        else:
            if save_masks or generate_figures:
                if not os.path.isdir('output_files'):
                    os.makedirs('output_files')
            if save_masks:
                mask_subdirectory = 'output_files/masks/'
                if not os.path.isdir(mask_subdirectory):
                    os.makedirs(mask_subdirectory)
                curr_name = mask_subdirectory + 'mask_fov_{}'.format(i + 1)
                i = 1
                if os.path.exists('{}_.tif'.format(curr_name)):
                    while os.path.exists('{}_({:d}).tif'.format(curr_name, i)):
                        i += 1
                    file_name = '{}_({:d}).tif'.format(curr_name, i)
                else:
                    file_name = '{}_.tif'.format(curr_name)
                mask_normalized = ((best_method - best_method.min()) * (1 / (best_method.max() - best_method.min()) * 255)).astype('uint8')
                tifffile.imwrite(file_name, mask_normalized)
            if include_cellpose and cellpose_data is None:
                raise ValueError('Cellpose data cannot be None! Please double check the data input')
            elif include_cellpose and cellpose_data is not None:
                curr_cellpose_image = cellpose_data[i]
            else:
                curr_cellpose_image = None
            comparison_images_subdirectory = 'output_files/comparison_images/'
            if not os.path.isdir(comparison_images_subdirectory):
                os.makedirs(comparison_images_subdirectory)
            generate_figures(image_dict,
                            best_method,
                            i,
                            saving_subdirectory=comparison_images_subdirectory,
                            display_figures=display_comparison_images,
                            cellpose=include_cellpose,
                            cellpose_image=curr_cellpose_image,
                            save_plots=save_comparison_images)
        best_method_list.append(best_method)
    if export_metrics_csv:
        csv_alt_export_driver(final_data_list, file_name_list, metric_list)

    print(total_time)
    return final_data_list, best_method_list


def pad_array(array):

    # Pad using standard parameters
    padded_array = np.pad(array, 
                          10, 
                          mode='constant',
                          constant_values=(0,0))
    return padded_array.astype(int)


def generate_figures(data, current_method, curr_image_number, saving_subdirectory='', display_figures=False,
                    cellpose=False, cellpose_image=None, save_plots=False):
    '''
    Wrapper figure for exporting figures; eventually integrate with dynamic plotting library.

    Parameters
    ----------
    data: dictionary
        Accumulator dictionary containing elements to plot (ex. composite image, mCherry photons, ground truth image, etc.)
    current_method: ndarray
        Image ndarray mask with cellprofiler style region labeling. Final output from `current_method`.
    curr_image_set: string
        String containing relevant information obout FOV/Dish; pulled directly from file name.
    save_plots: boolean, optional
        Boolean to save plots with transparent backgrounds.

    Returns
    -------
    None
    '''

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)
    if cellpose:
        fig, ax = plt.subplots(1, 3, figsize=(20, 11))
        #fig.suptitle(data.get('file_name'))
        ax[0].imshow(current_method, cmap=plt.cm.Greys_r)
        ax[0].set_title('Current method')
        ax[0].set_axis_off()
        ax[1].imshow(cellpose_image, cmap=plt.cm.Greys_r)
        ax[1].set_title('Cellpose')
        ax[1].set_axis_off()
        ax[2].imshow(data.get('gt'), cmap=plt.cm.Greys_r)
        ax[2].set_title('Ground truth')
        ax[2].set_axis_off()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 11))
        #fig.suptitle(data.get('file_name'))
        ax[0].imshow(current_method, cmap=plt.cm.Greys_r)
        ax[0].set_title('Current method')
        ax[0].set_axis_off()
        ax[1].imshow(data.get('gt'), cmap=plt.cm.Greys_r)
        ax[1].set_title('Ground truth')
        ax[1].set_axis_off()
    if cellpose and cellpose_image is not None:
        curr_name = data.get('file_name') + '_cellpose'
    else:
        curr_name = data.get('file_name')
    if save_plots:
        i = 1
        if os.path.exists(saving_subdirectory + '{}.png'.format(curr_name)):
            while os.path.exists(saving_subdirectory + '{}_({:d}).png'.format(curr_name, i)):
                i += 1
            file_name = saving_subdirectory + '{}_({:d}).png'.format(curr_name, i)
        else:
            file_name = saving_subdirectory + '{}.png'.format(curr_name)
        plt.savefig(file_name)
    if display_figures:
        plt.show()
    plt.close(fig)
    return True