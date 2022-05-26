from scipy.spatial.distance import directed_hausdorff
import numpy as np
import pandas as pd
import csv
import os


def calculate_dice(ground_truth, image):
    """
    Calculates the dice for a given image, provided a ground truth. Logic is implemented to require images of the same size.

    Parameters
    ----------
    ground_truth: np.ndarray
        Ground truth numpy ndarray image.
    image: np.ndarray
        Current image numpy ndarray.

    Returns
    -------
    dice_coef: int
        Dice coefficient score.
    """
    if ground_truth is None:
        raise ValueError('Ground truth is required to calculate dice coefficients')
    if not ground_truth.any() and not image.any():
        return 1
    if ground_truth.size != image.size:
        raise ValueError(
            'Size of the ground truth and calculated image must be the same! \n Currently, the ground is of size {} and the calculated image is of size {}.'.format(
                ground_truth.shape, image.shape))
    ground_truth = np.where(ground_truth > 0, 1, 0)
    image = np.where(image > 0, 1, 0)
    return (np.sum(image[ground_truth == 1]) * 2.0) / (np.sum(image) + np.sum(ground_truth))


def calculate_hausdorff(ground_truth, image):
    """
    Calculates the Hausdorff distance for a given image, provided a ground truth. Both arrays must have the same number of columns. 

    Parameters
    ----------
    gt: np.ndarray
        Ground truth numpy ndarray image.
    im: np.ndarray
        Current image numpy ndarray.

    Returns
    -------
    hausdorff_distance: tuple
        Hausdorff distance calculation; 3 values are returned: distance from gt to im, distance from im to gt, and .
    """
    if ground_truth is None:
        raise ValueError('Ground truth is required to calculate dice coefficients')
    return directed_hausdorff(image, ground_truth)


def calculate_jaccard(ground_truth, image):
    """
    Calculates the jaccard distance for a given image, provided a ground truth. Logic is implemented to require images of the same size.
    Jaccard index/distance is equivalent to the intersection over union (IOU). 

    Parameters
    ----------
    gt: np.ndarray
        Ground truth numpy ndarray image.
    im: np.ndarray
        Current image numpy ndarray.

    Returns
    -------
    jaccard_distance: float
        Calculated Jaccard index/distance.
    """
    if ground_truth is None:
        raise ValueError('Ground truth is required to calculate jaccard distance.')
    if not ground_truth.any() and not image.any():
        return 1
    if ground_truth.size != image.size:
        raise ValueError(
            'Size of the ground truth and calculated image must be the same! \n Currently, the ground is of size {} and the calculated image is of size {}.'.format(
                ground_truth.shape, image.shape))

    ground_truth = np.array(ground_truth, dtype=bool)
    image = np.array(image, dtype=bool)

    return (ground_truth * image).astype(int).sum() / float((ground_truth + image).astype(int).sum())


def check_dice(ground_truth, images):
    """
    Check dice: checks the ground truth against a series of mutliple calculated images and returns the dice coefficients for each.
    Parameters
    ----------
    ground_truth: np.ndarray
        Reference hand-segmented ground truth image to be compared against
    images: dictionary
        Dictionary of numpy ndarray's containing each of the computed images to be compared

    Examples
    --------
    test_gt = np.ones((512, 512))
    test_im = np.zeros((512, 512))
    test_im2 = np.ones((512, 512))

    dice_dict = check_dice(ground_truth = test_gt,
                            zeros = test_im,
                            ones = test_im2)
    for dice in dice_dict:
        print('Dice for method {} is {}'.format(dice, dice_dict.get(dice)))

    Returns
    -------

    """

    if ground_truth is None:
        raise ValueError('Ground truth is required to calculate dice coefficients')        
    dice_dict = dict()
    max_dice = 0
    best_method = ''
    best_image = None
    for method, image in images.items():
        curr_dice = calculate_dice(ground_truth, image)
        if curr_dice >= max_dice:
            max_dice = curr_dice
            best_method = method
            best_image = image
        dice_dict[method] = curr_dice
    return dice_dict, best_method, best_image


def check_hausdorff(ground_truth, images):
    """
    Check hausdorff: checks the ground truth against a series of mutliple calculated images and returns the hausdorff distance for each.
    Parameters
    ----------
    ground_truth: np.ndarray
        Reference hand-segmented ground truth image to be compared against
    images: dictionary
        Dictionary of n umpy ndarray's containing each of the computed images to be compared

    Examples
    --------
    test_gt = np.ones((512, 512))
    test_im = np.zeros((512, 512))
    test_im2 = np.ones((512, 512))

    hausdorff_dict = check_hausdorff(ground_truth = test_gt,
                            zeros = test_im,
                            ones = test_im2)
    for hausdorff_distance in hausdorff_dict:
        print('Hausdorff distance for method {} is {}'.format(hausdorff_distance, hausdorff_dict.get(dice)))

    Returns
    -------

    """

    if ground_truth is None:
        raise ValueError('Ground truth is required to calculate dice coefficients')
    if not ground_truth.any():
        raise ValueError('Ground truth cannot be entirely zeros!')
    hausdorff_dict = dict()
    for curr in images:
        hausdorff_dict[curr] = calculate_hausdorff(ground_truth, images.get(curr))
    return hausdorff_dict


def check_jaccard(ground_truth, images):
    """
    Check jaccard: checks the ground truth against a series of mutliple calculated images and returns the jaccard distance for each.

    Parameters
    ----------
    ground_truth: np.ndarray
        Reference hand-segmented ground truth image to be compared against
    images: dictionary
        Dictionary of n umpy ndarray's containing each of the computed images to be compared

    Examples
    --------
    test_gt = np.ones((512, 512))
    test_im = np.zeros((512, 512))
    test_im2 = np.ones((512, 512))

    jaccard_dict = check_jaccard(ground_truth = test_gt,
                            zeros = test_im,
                            ones = test_im2)
    for jaccard_value in jaccard_dict:
        print('Jaccard distance for method {} is {}'.format(jaccard_value, jaccard_dict.get(dice)))

    Returns
    -------

    """

    if ground_truth is None:
        raise ValueError('Ground truth is required to calculate dice coefficients')
    jaccard_dict = dict()
    for curr in images:
        jaccard_dict[curr] = calculate_jaccard(ground_truth, images.get(curr))
    return jaccard_dict


def calculate_image_statistics_helper(ground_truth, image_dictionary, *statistics_list):
    statistics_accumulator_dict = dict()
    for statistic in statistics_list:
        if statistic.lower() == 'Dice'.lower():
            dice_dict, best_method, best_image = check_dice(ground_truth=ground_truth, images=image_dictionary)
            statistics_accumulator_dict[statistic] = dice_dict
        elif statistic.lower() == 'Hausdorff'.lower():
            statistics_accumulator_dict[statistic] = check_hausdorff(ground_truth=ground_truth, images=image_dictionary)
        elif statistic.lower() == 'Jaccard'.lower():
            statistics_accumulator_dict[statistic] = check_jaccard(ground_truth=ground_truth, images=image_dictionary)
        else:
            raise ValueError('Incorrect image statistic selected, potential values include: {}'.format(statistics_list))
    return statistics_accumulator_dict, best_method, best_image




def csv_export_builder_alt(statistics_accumulator_dict_list, file_names, statistics_list):
    # CSV File Builder
    csv_dict_builder_list = list()
    for curr_dict_index, curr_data_dict in enumerate(statistics_accumulator_dict_list):
        for metric, combos in curr_data_dict.items():
            for combo in combos:
                temp_dict = dict()
                temp_dict['file_name'] = file_names[curr_dict_index]
                temp_dict['combo'] = combo
                temp_dict[metric] = curr_data_dict.get(metric).get(combo)
                csv_dict_builder_list.append(temp_dict)
    return csv_dict_builder_list




def csv_alt_export_driver(statistics_accumulator_dict_list, file_names, statistics_list):
    csv_file = "metrics_output"
    i = 1
    if os.path.exists('{}.csv'.format(csv_file)):
        while os.path.exists('{}_({:d}).csv'.format(csv_file, i)):
            i += 1
        file_name = '{}_({:d}).csv'.format(csv_file, i)
    else:
        file_name = '{}.csv'.format(csv_file)
    csv_columns = ['file_name', 'combo']
    csv_columns[1:1] = statistics_list
    csv_dict_builder_list = csv_export_builder_alt(statistics_accumulator_dict_list, file_names, statistics_list)
    csv_dict_builder_df = pd.DataFrame(csv_dict_builder_list)
    print(csv_dict_builder_df)
    csv_dict_builder_df_combined = pd.concat([curr_csv_dict_df.apply(lambda x: sorted(x, key=pd.isnull)) for _, curr_csv_dict_df in csv_dict_builder_df.groupby('file_name', sort=False)]).dropna(subset=statistics_list)
    try:
        csv_dict_builder_df_combined.to_csv(file_name, index=False)
    except IOError:
        print("I/O error")