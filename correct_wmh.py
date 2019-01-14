#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm that implements statistically motivated corrections to the white matter
Hyperintensity (WMH) maps of a single subject collected at two different visits.
"""

import os
import csv
import glob
import numpy as np
import nibabel as nib    # Python neuroimaging file routines
from scipy import ndimage # N-dimensional image processing



# Helper function to create a binary mask with a one in each voxel where the
# WMH score is above the threshold 'thr'
def create_threshold_mask(thr, wmh_map):
    
    wmh_mask = np.zeros(wmh_map.shape)
    
    above_thr = wmh_map > thr

    wmh_mask[above_thr] = 1
            
    return wmh_mask



# Helper function to calculate the Dice coefficient directly from the WHM masks
def dice(mask_1, mask_2):
    
    tot_mask = mask_1 + mask_2
    
    num_wmh_agree = tot_mask[tot_mask == 2].size
    
    num_wmh_disagree = tot_mask[tot_mask == 1].size
    
    dice_coeff = 2*num_wmh_agree/(2*num_wmh_agree + num_wmh_disagree)

    return dice_coeff  



def correct_wmh(thr, maps, masks):
    
    # Combining the two masks in this way yields:  
    #   '1' where voxel is marked WMH in first mask only,
    #   '2' where voxel is marked WMH in second mask only,
    #   '3' where voxel is marked WMH in both masks, 
    #   and zero elsewhere
    region_mask = masks[0] + 2*masks[1]
    
    wmh_overlap = (region_mask == 3)
    
    # Find the means of the two maps in the region in which they agree
    mean_1 = maps[0][wmh_overlap].mean()
    mean_2 = maps[1][wmh_overlap].mean()
        
    # Calculate new baseline WMH threshold using a simple linear model
    base_thr = 0.6*min(mean_1, mean_2) - 0.3
    
    # Constrain base_thr to fall between 1.5 and 3.0 standard deviations
    if base_thr < 1.5:        
        base_thr = 1.5
    elif base_thr > 3.0:
        base_thr = 3.0
        
    diff = abs(mean_1 - mean_2)

    # If the means of the two scans are widely different (diff >> 0), then
    # use base_thr, as previous studies have shown it to be a reliable alternative
    # in such cases.
    new_thr = max(base_thr, thr - diff)
        
    if mean_1 > mean_2:
        better_visit = 0
        worse_visit  = 1
    else:
        better_visit = 1
        worse_visit  = 0
        
    better_mask = masks[better_visit].copy()
    worse_map = maps[worse_visit].copy()
    worse_mask = masks[worse_visit].copy()
    better_region = better_visit + 1
    
    new_worse_mask = worse_mask.copy()

    # Rethreshold the scores_to_revisit (those from worse_map that were tagged as
    # WMH in the higher quality mri scan) at new_thr to create a corrected WMH mask
    # for the lower quality scan       
    if new_thr < thr:
        
        better_region_mask = np.zeros(region_mask.shape)
    
        better_region_mask[region_mask == better_region] = 1
    
        scores_to_revisit = better_region_mask*worse_map
     
        new_worse_mask[scores_to_revisit > new_thr] = 1

    # Corrections to higher quality mask 
    
    # Recalculate region_mask using the results of the low quality image corrections
    new_region_mask = better_mask + 2*new_worse_mask
    new_wmh_overlap = (new_region_mask == 3)  
    
    # Find the revised region of agreement (i.e. voxels tagged as WMH in both visits)
    overlap_region_mask = np.zeros(new_region_mask.shape)
    overlap_region_mask[new_wmh_overlap] = 1
    
    # Dilate the region of agreement by one voxel
    dilated_mask = ndimage.morphology.binary_dilation(overlap_region_mask, iterations=1)
  
    # Subtracting the two masks yields a one-voxel thick boundary mask
    # (Width of one voxel found to be optimal in previous study)
    border_mask = dilated_mask - overlap_region_mask

    # If voxels in boundary_mask were tagged as WMH in the lower quality scan,
    # add them to the higher quality mask     
    worse_region_mask = np.zeros(new_region_mask.shape)
    worse_region_mask[new_region_mask == 2] = 1       
    vox_to_add = border_mask*worse_region_mask    
    new_better_mask = better_mask.copy()        
    new_better_mask[vox_to_add > 0] = 1
      
    if better_visit == 0:       
        return [new_better_mask, new_worse_mask]   
    else:     
        return [new_worse_mask, new_better_mask]
    
        
def mask_comparison_stats(mask_1, mask_2):
    # Multiply the binary masks and sum to get the number of agreed upon voxels
    num_common_voxels = int((mask_1*mask_2).sum())

    # Use the difference of the two masks to count how many voxels were tagged as WMH
    # only in visit #1 and visit #2, respectively
    diff_mask = mask_1 - mask_2
    num_only_tagged_in_v1 = (diff_mask > 0).sum()
    num_only_tagged_in_v2 = (diff_mask < 0).sum()  

    return [num_common_voxels, num_only_tagged_in_v1, num_only_tagged_in_v2]


def save_corrected_mask(filename, mask, nii):
        mask_nii = nib.nifti1.Nifti1Image(mask, None, header=nii.header.copy())    
        nib.save(mask_nii, filename)


def main(subject_dir, visit_dirs, stats_file):
    
    # Optimal threshold as determined by UCSD group
    thr = 3.0

    start_dir = os.getcwd()    
    os.chdir(subject_dir)
    
    # Write header line to output 'stats_file'
    with open(subject_dir + stats_file, 'w') as csvfile:
        
        csvwriter = csv.writer(csvfile, delimiter='\t')
        csv_header = ['Subject_Date_IDNUM', 'WMH_1', 'WMH_2', 'Common', 'Only_1', 'Only_2', 'Dice']
        csvwriter.writerow(csv_header)

    # visit_dirs is a list of subdirectories of subject_dir     
    for i in range(len(visit_dirs)):
        
        # Here we are using the directory structure specified by the Mind Research Network
        cur_dir = subject_dir + visit_dirs[i] + 'longitudinal_ac/'    
        os.chdir(cur_dir)    
        subject = cur_dir.split('/')[-4]    
        study_date = cur_dir.split('/')[-3][5:13]        
        file_id = subject + '_' + study_date                   
        
        # Inputs are the Z-score files produced by the UCSD WMH segmentation algorithm
        wmh_map_files = sorted(glob.glob('*_wmh_Zscore_TP.nii.gz'))
        print(wmh_map_files)
    
        # Load the images and extract the maps
        v1_nii = nib.load(wmh_map_files[0])
        v2_nii = nib.load(wmh_map_files[1])
    
        wmh_map_1 = v1_nii.get_data().astype(float)
        wmh_map_2 = v2_nii.get_data().astype(float)
    
        # Create the WMH masks using the input threshold
        orig_mask_1 = create_threshold_mask(thr, wmh_map_1)
        orig_mask_2 = create_threshold_mask(thr, wmh_map_2)
    
        # Apply correction subroutine
        new_masks = correct_wmh(thr, [wmh_map_1, wmh_map_2], [orig_mask_1, orig_mask_2])

        # Dice coefficient measures 'agreement' between two WMH masks
        orig_dice = dice(orig_mask_1, orig_mask_2)    
        print('Dice coeff for uncorrected masks:  ' + str(orig_dice))
    
        new_dice = dice(new_masks[0], new_masks[1])    
        print('Dice coeff for corrected masks:  ' + str(new_dice))
        print()

        # Save the corrected WMH masks as nifti files
        save_corrected_mask(cur_dir + 'v1_corrected.nii.gz', new_masks[0], v1_nii)
        save_corrected_mask(cur_dir + 'v2_corrected.nii.gz', new_masks[1], v2_nii) 
        
        # Calculate and save the WMH change mask that results from the above corrections being applied
        change_mask = new_masks[1] - new_masks[0]    
        change_mask_nii = nib.nifti1.Nifti1Image(change_mask, None, header=v2_nii.header.copy())        
        nib.save(change_mask_nii, cur_dir + 'change_mask.nii.gz')       
    
        # Calculate before and after correction stats
        [orig_num_common_vox, orig_num_only_v1, orig_num_only_v2] = mask_comparison_stats(orig_mask_1, orig_mask_2)
        [new_num_common_vox, new_num_only_v1, new_num_only_v2] = mask_comparison_stats(new_masks[0], new_masks[1])
    
        orig_wmh_totals = [file_id, int(orig_mask_1.sum()), int(orig_mask_2.sum()), orig_num_common_vox, orig_num_only_v1, orig_num_only_v2, orig_dice]    
        new_wmh_totals  = [file_id, int(new_masks[0].sum()), int(new_masks[1].sum()), new_num_common_vox, new_num_only_v1, new_num_only_v2, new_dice]
    
        # Save statistics to 'stats_file' in top-level subject directory
        with open(subject_dir + stats_file, 'a') as csvfile:
        
            csvwriter = csv.writer(csvfile, delimiter='\t')
        
            csvwriter.writerow(orig_wmh_totals)
            csvwriter.writerow(new_wmh_totals)
          
    os.chdir(start_dir)

    
main()    
    
    