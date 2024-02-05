# MRI-Tools

## bi_tensor.py 
- My implementation of the free water diffusion-tensor imaging (DTI) algorithm described in the paper:

  "Free water elimination and mapping from diffusion MRI"
  by Ofer Pasternak,  Nir Sochen,  Yaniv Gur,  Nathan Intrator,  Yaniv Assaf,
  First published: 21 July 2009 https://doi.org/10.1002/mrm.22055

The algorithm essentially breaks the Diffusion Tensor Imaging (DTI) MRI signal into two components (modeling two different aspects of water diffusion in the brain) as a bi-tensor.  Gradient descent is then used to extract optimal diffusion parameters using the forward function applied to data from both normal and diseased tissue.  The specific data structure I chose (conversion of a stack of 3-vectors (v), returns a stack of 3x3 diagonal matrices w/ the 'v' on diagonals) enables efficient numerical solution because it becomes possible to vectorize the eigenvalue constraints before applying gradient descent.

## correct_wmh.py 
- An algorithm I designed that implements statistically-motivated corrections to the White Matter Hyperintensity (WMH) maps of a single subject collected via MRI scans at two different visits.  WMH maps are one component in the diagnosis of many tissue-related changes in the brain, but may also be artifacts of the imaging process.  Comparing the results of scans on the same individual separated in time can help to determine disease progression as well as to eliminate potential artifacts (especially when seen in an earlier scan, but not in a later one) or seen in a low quality image but not at higher resolution.

I first create binary maps denoting where WMH voxels are detected for all relevant image files.  The DICE coefficient is the main statistic capturing the relationship between any two masks, while the standard deviations are used to generate WMH intensity thresholds for subsequently revisiting questionable voxels.  Identification of voxels determined to be misclassified in earlier scans can be used to generate an improved threshold and reanalyze the historical data.  This approach can be repeated with each new scan at a later timepoint, resulting in the continual improvement of historical data and the ability to track disease progression.

## wmh_correction_reproducibility.ipynb 
- A Jupyter notebook that shows how a boundary mask width of one was determined to be optimal for one of the corrections implemented in correct_wmh.py above.

One key statistical component of the aporoach outlined above in correct_wmh.py is the identification of the optimal width to use to mask the boundaries of isolated brain regions containing contiguous WMH voxels (potential lesions).  The basic approach is to calculate the dice coefficients for different dilation sizes and look for a 'kink' in the graph which marks the point of diminishing returns as far as reproducibility goes, i.e., the dilation size at which further improvement in the dice coefficient falls off precipitously.  As can be seen from the plot at the end of the Jupyter notebook, this transition reliably takes place for a dilation size of one voxel.  It is this value that is used as a key parameter to the algorithm described in correct_wmh.py above.

## wmh_statistics.py 
- This file contains several Python helper functions designed to calculate WMH segmentation performance statistics that can aid in the evaluation and comparison of different algorithmic approaches to the segmentation problem.

### Contents
- find_mask_errors:  A function that finds the number of false positive and false negative voxels mistakenly identified by an input mask, using a manually segmented mask as the ground truth.
- find_true_positives:  A function that finds the number of true positive voxels identified by trial_mask, using manual_mask as the ground truth.
- calculate_dice:  A function that calculates the Dice similarity coefficient between two binary masks.
