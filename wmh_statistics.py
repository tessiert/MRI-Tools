#!/usr/bin/env python
"""Python script to calculate WMH segmentation performance statistics

Calculate # of voxels correctly & incorrectly modified (and corresponding Dice coefficient) by
	Single-visit optimization stage
	Longitudinal correction stage
	Entire algorithm"""

import os
import sys
import pathlib
import datetime
import subprocess
import shutil



# Function that finds the number of false positive and false negative voxels mistakenly identified by trial_mask, using manual_mask as the ground truth
def find_mask_errors(tmp_dir, trial_mask, manual_mask):

    # name of image file to hold the results of mask subtraction
    mask_diff = tmp_dir + "mask_diff"
    
    # Find difference between uncorrected auto-segmented mask and manual mask
    subprocess.call(["fslmaths", trial_mask, "-sub", manual_mask, mask_diff])

    # A voxel value of +1 => false positive
    fcn_call = ["fslstats", mask_diff, "-l", "0", "-V"]
    output = subprocess.check_output(fcn_call, universal_newlines=True)
    num_false_pos = int(output.split()[0])

    # A voxel value of -1 => false negative
    fcn_call = ["fslstats", mask_diff, "-u", "0", "-V"]
    output = subprocess.check_output(fcn_call, universal_newlines=True)
    num_false_neg = int(output.split()[0])

    return(num_false_pos, num_false_neg)



# Function that finds the number of true positive voxels identified by trial_mask, using manual_mask as the ground truth
def find_true_positives(tmp_dir, trial_mask, manual_mask):

    # name of image file to hold the results of mask addition
    mask_sum = tmp_dir + "mask_sum"

    # Take sum of uncorrected auto-segmented mask and manual mask
    subprocess.call(["fslmaths", trial_mask, "-add", manual_mask, mask_sum])

    # A voxel value of +2 => true positive
    fcn_call = ["fslstats", mask_sum, "-l", "1.9", "-V"]
    output = subprocess.check_output(fcn_call, universal_newlines=True)
    num_true_pos = int(output.split()[0])

    return(num_true_pos)



# Function that calculates the Dice similarity coefficient
def calculate_dice(num_true_pos, num_false_pos, num_false_neg):

    dice = 2*num_true_pos/(2*num_true_pos + num_false_pos + num_false_neg)

    return(dice)



# Function that calculates classification statistics for the two input WMH masks
def calculate_stats(tmp_dir, mask_1, mask_2):

    (num_false_pos, num_false_neg) = find_mask_errors(tmp_dir, mask_1, mask_2) 
 
    num_true_pos = find_true_positives(tmp_dir, mask_1, mask_2)

    dice = calculate_dice(num_true_pos, num_false_pos, num_false_neg) 

    print(mask_1.split('/')[-1].split('.')[0] + ":")
    print("# False Positives = %7i\t# False Negatives = %7i\t# True Positives = %7i\tDice Coef. = %.3f\n" %(num_false_pos, num_false_neg, num_true_pos, dice))
    


# Argument processing, initializations, and execution
def main():

    # Process arguments (sys.argv[0] is name of python script)
    auto_seg_mask = sys.argv[1]
    manual_mask = sys.argv[2]
    opt_mask = sys.argv[3]
    final_mask = sys.argv[4]

    # Make timestamped temp working directory
    cur_dir = os.getcwd()
    now = datetime.datetime.now()
    time_stamp = ".wmh_stats_" + str(now.month) + str(now.day) + str(now.year) + str(now.hour) + str(now.minute) + str(now.second)
    tmp_dir = cur_dir + "/" + time_stamp + "/"
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True) 

    # Compare initial segmentation mask to ground truth mask
    calculate_stats(tmp_dir, auto_seg_mask, manual_mask)

    # Compare single_visit optimized mask to ground truth mask
    calculate_stats(tmp_dir, opt_mask, manual_mask)

    # Compare longitudinally corrected mask to ground truth mask
    calculate_stats(tmp_dir, final_mask, manual_mask)

    # Separator to make piped output from successive runs more readable
    print("-----------------\n")

    # Clean up
    shutil.rmtree(tmp_dir)


# Execute
main()




