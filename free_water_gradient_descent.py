
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti



# Constant declarations
d = 3*1e-3       # diffusivity of free water
b = 1000         # single-shell b-value
A_w = np.exp(-b*d)  # free water attenuation



# Helper function to do stacked matrix manipulation
def vec_mat_vec_prod(v, M):
    
    # Product M*v (i, j - dimensions of M, k - # of gradient directions => result is one 3-vec for each grad. dir.)
    temp = np.einsum('...ij,kj->...ik', M, v)
    
    # Product v_T*M*v (result is one value for each grad. dir.)
    return np.einsum('...ki,...ik->...k', v, temp)



# Bi-Tensor Model Forward Function (Returns signal attenuation A, and optionally S0, at a given voxel)
def fwd_model(b_vecs, D, f, S0=None):
    
    # Calculate exponent for applied gradient direction == q_i'*D*q_i for arbitrary number of input b_vecs
    D_i = np.moveaxis(vec_mat_vec_prod(b_vecs, D), 3, 0)
    
    # Calculate attenuated signal
    A_bi = np.moveaxis(f*np.exp(-b*D_i) + (1-f)*np.exp(-b*d), 0, 3)
    
    # If S0 provided, also return raw signal
    if S0 is not None:
    
        S_bi = S0*A_bi
        return S_bi, A_bi
    
    else:
    
        return A_bi
    

    
# Helper function:   
# Let Ei be the n×n matrix with a 1 on position (i,i) and zeros everywhere else; 
# similarly, let ei be the 1×n row matrix with a 1 on position (1,i) and zeros everywhere else. Then
# D = ∑(Ei*v*ei), where v is vector of eigenvalues, and D is corresponding diagonal matrix 

# Function that, given a stack of 3-vectors (v), returns a stack of 3x3 diagonal matrices w/ the 'v' on diagonals
# (Allows for vectorization of eigenvalue constraints in grad_descent
def stack_diags(evals):

    E1 = np.zeros(D0.shape)
    E2 = np.zeros(D0.shape)
    E3 = np.zeros(D0.shape)
    
    E1[...,0,0] = 1
    E2[...,1,1] = 1
    E3[...,2,2] = 1
    
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    
    temp = np.moveaxis(evals*e1, 3, 0)
    
    diag_stack = np.moveaxis(np.moveaxis(E1, [3,4], [0,1])*temp, [0,1], [3,4])
    
    temp = np.moveaxis(evals*e2, 3, 0)
    
    diag_stack = diag_stack + np.moveaxis(np.moveaxis(E2, [3,4], [0,1])*temp, [0,1], [3,4])
    
    temp = np.moveaxis(evals*e3, 3, 0)
    
    diag_stack = diag_stack + np.moveaxis(np.moveaxis(E3, [3,4], [0,1])*temp, [0,1], [3,4])
    
    return diag_stack 




# The gradient descent algorithm:

"""
Notes:  
- In Pasternak 2009, evals of D are constrained to lie between 0.1e-3 and 2.5e-3.  These constraints are the defaults of 
the adjustable parameters ('lambda_min' and 'lambda_max') of grad_descent.

- Pasternak et. al., also take D0 evals (< 0) or (> d) out of the fit by default.  This is the default of the boolean 
flag 'restrict_vox'.

    - Turning 'restrict_vox' off will allow the tissue fraction values 'f' in regions of potential flow, e.g., in the csf,
    to participate in the fit - typically resulting in increased structure in these regions in the free water and tissue 
    fraction maps.
    
- The boolean flag 'display_outliers' will determine whether or not the extreme D0 evals will be reinserted into d_fit.  
A value of True will typically yield greater structure in high water content regions in the FA and MD maps.
"""
def grad_descent(D0, f0, A_meas, bvecs, alpha=1, beta=1, num_iter=100, step_size=1e-7, tol=1e-6, lambda_min=0.1e-3, lambda_max=2.5e-3,
                 restrict_vox=True, display_outliers=True):

    # Find min and max evals of D0 at each voxel
    D0_evals = np.linalg.eigvalsh(D0)
    min_evals = D0_evals.min(axis=3)
    max_evals = D0_evals.max(axis = 3)
    
    # Working copy of brain_mask
    brain_mask_mod = brain_mask.copy()
    
    # If user wants to take D0 evals < 0 or > d out of the running
    if restrict_vox:
            
        # Remove voxels w/ evals > d (flow effects) or < 0 from brain_mask, so they don't affect the fit
        brain_mask_mod[min_evals < 0] = 0
        brain_mask_mod[max_evals > d] = 0
    
    # Modified brain mask repeated # gradient directions number of times
    rep_mask = np.repeat(brain_mask_mod[..., np.newaxis], A_meas.shape[3], axis=3)
    
    # Define constant valued partials of D wrt first 6 elements of X (X defined just above Pasternak, 2012, eq.6)
    p_DX1 = np.array([[1,0,0],[0,0,0],[0,0,0]])
    p_DX2 = np.array([[0,0,0],[0,1,0],[0,0,0]])
    p_DX3 = np.array([[0,0,0],[0,0,0],[0,0,1]])
    p_DX4 = 1/np.sqrt(2) * np.array([[0,1,0],[1,0,0],[0,0,0]])
    p_DX5 = 1/np.sqrt(2) * np.array([[0,0,0],[0,0,1],[0,1,0]])
    p_DX6 = 1/np.sqrt(2) * np.array([[0,0,1],[0,0,0],[1,0,0]])
    
    # 6 x 3 x 3 array of partials
    pD = np.array([p_DX1, p_DX2, p_DX3, p_DX4, p_DX5, p_DX6])
    
    # Starting values for fit
    D_cur = D0.copy()
    f_cur = f0.copy()*brain_mask_mod
    
    # Set background of measured attenuations to zero
    A_meas_no_bkg = A_meas.copy()*rep_mask
    
    # Keep track of evolution of cost function
    costs = []
    
    # Track number of iterations
    count = 0
    
    # Start value for fractional change
    change = 1/tol

    #for n in range(num_iter):
    while (count < num_iter) and (change > tol):
        
        # First six indices (D_j_t) in def. of X just above Pasternak, 2012, eq.6 are captured by the last two indices
        # of D_cur, while the last 3 indices of X correspond to the (x,y,z) voxel coords.  But voxel coords coorespond 
        # to the first three indices of D_cur, so need only six entries in X array.
        X = np.array([D_cur[:,:,:,0,0], D_cur[:,:,:,1,1], D_cur[:,:,:,2,2], np.sqrt(2)*D_cur[:,:,:,0,1], 
                     np.sqrt(2)*D_cur[:,:,:,1,2], np.sqrt(2)*D_cur[:,:,:,0,2]])
        
        # Reorder indices for consistency w/ order required by det(gamma) below => dim[x] = dim_x x dim_y x dim_z X 6
        X = np.moveaxis(X, 0, 3)
        
        # Construct spatial partial derivatives of the first six entries in the vector X
        # These entries are of the form D_ij, so use symmetric finite differences
        grad_X = np.gradient(X, axis=(0, 1, 2))
        px_X = grad_X[0].copy()
        py_X = grad_X[1].copy()
        pz_X = grad_X[2].copy()
        
        # Calculate six independent entries of gamma matrix
        # (see notes for expansion of each term out of Einstein summation notation)
        gamma_xx = 1 + beta * (px_X**2).sum(axis=3)
        gamma_yy = 1 + beta * (py_X**2).sum(axis=3)
        gamma_zz = 1 + beta * (pz_X**2).sum(axis=3)
        gamma_xy = beta * (px_X * py_X).sum(axis=3)
        gamma_yz = beta * (py_X * pz_X).sum(axis=3)
        gamma_xz = beta * (px_X * pz_X).sum(axis=3)
        
        # Create (symmetric) gamma matrices (one for each voxel)
        gamma = np.array([[gamma_xx, gamma_xy, gamma_xz], [gamma_xy, gamma_yy, gamma_yz], [gamma_xz, gamma_yz, gamma_zz]])
        
        # Reorder indices so gamma indices come last (needed by np.linalg.det)
        gamma = np.moveaxis(gamma, [0, 1], [3, 4])
        
        # Calculate det(gamma)
        det_gamma = np.linalg.det(gamma)
        
        # Calculate inverse of gamma
        gamma_inv = np.linalg.inv(gamma)
        
        # Full bi-tensor forward model values (one for each gradient direction) for current weights 'f'
        A_bi = fwd_model(bvecs, D_cur, f_cur)
        
        # Forward model with f == 1 gives single (tissue) compartment values
        A_tissue = fwd_model(bvecs, D_cur, 1)
        
        # Don't let attenuations stray outside biologically meaningful bounds
        np.clip(A_bi, 0, 1, out=A_bi) 
        np.clip(A_tissue, 0, 1, out=A_tissue)
        
        # Apply brain mask to results for each gradient direction
        A_bi = A_bi * rep_mask
        A_tissue = A_tissue * rep_mask
        
        # Terms of the form q_i'*pD*q_i (one for each 'j')
        q_pD_q = vec_mat_vec_prod(bvecs, pD)
        
        # Calculate Eq. 6 of Pasternak, 2012 (see notes for expansions out of Einstein summation notation)
        
        # Array to hold independent entries of update rule for D
        D_t = []
        
        # For each independent entry of D_t
        for j in range(6):
            
            # Sum over 'i' to get first term of eq. 6
            D_1st = b*((A_bi - A_meas_no_bkg)*A_tissue*q_pD_q[j]).sum(axis=3)
            
            # Calculate 2nd term in eq. 6
            
            # Gradient of jth component of X
            grad_X_j = np.moveaxis(np.array([px_X[:,:,:,j], py_X[:,:,:,j], pz_X[:,:,:,j]]), 0, 3)
                
            # See http://ajcr.net/Basic-guide-to-einsum/ for a refresher on how to work with einsum
                
            # First moveaxis needed because multiplication by det_gamma proceeds by broadcasting first axis of
            # einsum result.  Second moveaxis returns result to normal ordering
            temp = np.moveaxis(np.sqrt(det_gamma)
                                   *np.moveaxis(np.einsum('...ij,...j->...i', gamma_inv, grad_X_j), 3, 0), 0, 3)               
                
            # Sum over spatial partials (finite diffs) of intermediate result above
            # Note:  Since the 'nu' index is summed over in the above matrix multiplication, the resulting
            # length three vectors (one at each voxel) are left with a 'mu' index, so take the gradient
            # wrt, and sum over, this index
            temp2 = np.array(np.gradient(temp, axis=3)).sum(axis=3)
                
            # 2nd term for current 'j'
            D_2nd = alpha/np.sqrt(det_gamma)*temp2*brain_mask_mod 
            
            # Next, calculate jth entry of D_t
            D_t.append(D_1st + D_2nd)
            
        # Form the symmetric update matrix (list entries:  0 = xx, 1 = yy, 2 = zz, 3 = xy, 4 = yz, 5 = xz)
        delta_mat = np.moveaxis(np.array([[D_t[0], D_t[3], D_t[5]], 
                                          [D_t[3], D_t[1], D_t[4]],
                                          [D_t[5], D_t[4], D_t[2]]]), [0,1], [3,4])
        
        # Apply the updates to find D_cur
        D_cur = D_cur + step_size*delta_mat
        
        # Restrict eigenvalues of D_cur to lie between 0.1e-3 and 2.5e-3
        # (From 'Iterations' section under eq. 7 in Pasternak, 2009)
        
        # First find eigensystem
        evals, Q = np.linalg.eigh(D_cur)
        
        # Apply constraints on evals
        evals[evals < lambda_min] = lambda_min
        evals[evals > lambda_max] = lambda_max
        
        # For real, symmetric matrix, D_cur = Q*L_diag*Q')
        L_diag = stack_diags(evals)
        D_cur = np.matmul(Q, np.matmul(L_diag, Q.transpose((0,1,2,4,3)))) # Transpose only last two dimensions of Q
                
        # Update rule for f (eq. 7)
        delta_f = -b*((A_bi - A_meas_no_bkg)*(A_tissue - A_w*rep_mask)).sum(axis=3)      
        f_cur = f_cur + step_size*delta_f
        
        # Enforce constraints on any f-values that have gone out of bounds
        f_cur[f_cur < 0] = 0
        f_cur[f_cur > 1] = 1
        
        # Calculate current value of cost function
        
        # Euclidean distance between model and measured values (per voxel)
        A_diff = np.linalg.norm(A_bi - A_meas_no_bkg, axis=3)
        
        # Beltrami constraint (per voxel)
        beltrami = alpha*np.sqrt(det_gamma)*brain_mask_mod
        
        # Maintain a record of cost function evolution
        costs.append((A_diff + beltrami).sum())
        #print((A_diff + beltrami).sum())
        
        # Update stop criteria
        count += 1
        
        # If this is not the first iteration
        if count != 1:
            change = abs(costs[-2] - costs[-1])/costs[-2]
    
    # Reinsert 'out of bounds' single-compartment DTI voxels?
    # This is done independently of value of restrict_vox since these may still have been thresholded out depending
    # on the values of lambda_min and lambda_max
    if display_outliers:
        D_cur[min_evals < 0] = D0[min_evals < 0]
        D_cur[max_evals > d] = D0[max_evals > d]
        
    # Display convergence info
    print('Number of iterations:  {}'.format(count))
    print('Last fractional change:  {}'.format(change))
        
    return D_cur, f_cur, costs




# Main driver: process all visits listed in study_dirs    
start_dir = os.getcwd()

main_dir = '/export/research/analysis/human/grosenberg/ugrant_20294/diffusion_simulation/UCD_comparison/'
archive_dir = main_dir + 'archive/'

study_dirs = ['M87101112/Study20171129at155709/', 'M87114828/Study20170920at103412/', 'M87123537/Study20170825at105126/', 
              'M87136542/Study20171215at131150/', 'M87139362/Study20170901at103638/', 'M87143787/Study20140905at113810/',
              'M87143787/Study20150925at143709/', 'M87198610/Study20170825at133443/']

# Make output directories if they don't exist
out_dirs = [main_dir + 'vary_d/' + cur_study for cur_study in study_dirs]

for cur_out_dir in out_dirs:
    if not os.path.exists(cur_out_dir):
        os.makedirs(cur_out_dir)

# Create full blown paths to data
data_dirs = [archive_dir + cur_study + 'sb1000/' for cur_study in study_dirs]

# Run the gradient descent method on each visit
for idx, cur_dir in enumerate(data_dirs):
    
    # Above, we constructed one output directory for each visit
    cur_out_dir = out_dirs[idx]
    
    # Change into current directory
    os.chdir(cur_dir)
    
    # Load the dti data
    nii = nib.load('dti_1000.nii.gz')
    img_data = nii.get_data()
    bvals, bvecs = read_bvals_bvecs('bvals_1000', 'bvecs_1000')
    
    # Extract the data corresponding to b = 0 or b = 1000
    bvals_eq_0_1000 = (bvals == 0) | (bvals == 1000)
    new_bvals = bvals[bvals_eq_0_1000].copy()
    new_bvecs = bvecs[bvals_eq_0_1000].copy()
    new_img_data = img_data[:,:,:,bvals_eq_0_1000].copy()
    
    # Construct the gradient table
    grad_table = gradient_table(new_bvals, new_bvecs)
    
    # Extract brain so we don't fit the background
    brain_img_data, brain_mask = median_otsu(new_img_data, 2, 1) 

    # Save brain and mask images
    mask_img = nib.Nifti1Image(brain_mask.astype(np.float32), nii.affine)
    brain_img = nib.Nifti1Image(brain_img_data.astype(np.float32), nii.affine)

    nib.save(mask_img, cur_out_dir + 'dti_brain_mask.nii.gz')
    nib.save(brain_img, cur_out_dir + 'dti_brain_img.nii.gz')

    # Run the single-compartment dti model and fit
    ten_model = dti.TensorModel(grad_table)
    ten_fit = ten_model.fit(brain_img_data)
    
    # Set up to fit the bi-tensor model
    
    # Extract b0 images
    bvals_eq_0 = (bvals == 0)

    b0_vals = bvals[bvals_eq_0].copy()
    b0_vecs = bvecs[bvals_eq_0].copy()
    b0_img_data = img_data[:,:,:,bvals_eq_0].copy()

    # Apply the brain mask
    for i in range(b0_img_data.shape[3]):
        b0_img_data[:,:,:,i] = b0_img_data[:,:,:,i]*brain_mask
    
    # Save the b0 image
    b0_header = nii.header.copy()
    b0_nii = nib.Nifti1Image(b0_img_data, nii.affine, header = b0_header)
    nib.save(b0_nii, cur_out_dir + 'dti_b0_image.nii.gz') 

    # Use average of b0 images for S0
    S0 = b0_img_data.mean(axis=3).astype(float, copy=True)
    
    # Make sure everything is positive
    # Zeros not allowed since we are taking log below
    S0[S0 <= 0] = dti.MIN_POSITIVE_SIGNAL
    
    # Extract b = 1000 images
    b_eq_1000 = (bvals == 1000)

    # 'ss' => single shell
    ss_bvals = bvals[b_eq_1000].copy()
    ss_bvecs = bvecs[b_eq_1000].copy()
    ss_img_data = img_data[:,:,:,b_eq_1000].astype(float, copy = True)

    # Calculate measured attenuations (S/S0)
    A_meas = np.zeros(ss_img_data.shape)

    for i in range(ss_img_data.shape[3]):
    
        cur_img = ss_img_data[:,:,:,i]
    
        # Normalize
        A_meas[:,:,:,i] = cur_img/S0
    
        # Constrain attenuations
        A_meas[A_meas[:,:,:,i] < 0] = 0
        A_meas[A_meas[:,:,:,i] > 1] = 1
    
    # Extract the 3 x 3 diffusion tensors (D) for each voxel, use these as D_0 values for the fit
    D0 = ten_fit.quadratic_form

    # ******** New way of initializing f0 (uses single compartment MD) *********** 
    MD_sc = dti.mean_diffusivity(ten_fit.evals)
    MD_img = nib.Nifti1Image(MD_sc.astype(np.float32), nii.affine)
    nib.save(MD_img, cur_out_dir + 'MD_sc_img.nii.gz')
    
    f0 = 1 - (MD_sc - MD_sc.min())/(MD_sc.max() - MD_sc.min())
    
    # First pass with alpha = 1
    D_fit_1, f_fit_1, costs_1 = grad_descent(D0, f0, A_meas, ss_bvecs, alpha=1, beta=1, num_iter=10, step_size=1e-7, 
                                             tol=1e-6, lambda_min=0.1e-3, lambda_max=2.5e-3, restrict_vox=False, 
                                             display_outliers=False)  
    
    # Now do 2nd pass with alpha turned off
    #    * As in 'Iterations' section of Pasternak, 2009 under eq. 7
    D_fit_2, f_fit_2, costs_2 = grad_descent(D_fit_1, f_fit_1, A_meas, ss_bvecs, alpha=0, beta=1, num_iter=10, 
                                             step_size=1e-7, tol=1e-6, lambda_min=0.1e-3, lambda_max=2.5e-3, 
                                             restrict_vox=False, display_outliers=False)        
    
    # Save results
    
    # Free water
    free_water = (1 - f_fit_2)*brain_mask
    FW_img = nib.Nifti1Image(free_water.astype(np.float32), nii.affine)
    nib.save(FW_img, cur_out_dir + 'FW_img.nii.gz')   
    
    # Eigenvalues of diffusion tensor (bi-compartment model)
    bi_evals = np.linalg.eigvalsh(D_fit_2)
    
    # FA 
    FA_bi = dti.fractional_anisotropy(bi_evals)*brain_mask
    FA_img = nib.Nifti1Image(FA_bi.astype(np.float32), nii.affine)
    nib.save(FA_img, cur_out_dir + 'FA_img.nii.gz')  
    
    # MD
    MD_bi = dti.mean_diffusivity(bi_evals)
    MD_img = nib.Nifti1Image(MD_bi.astype(np.float32), nii.affine)
    nib.save(MD_img, cur_out_dir + 'MD_img.nii.gz')
    
    # Residual maps 
    
    # Modified brain mask repeated # gradient directions number of times
    rep_mask = np.repeat(brain_mask[..., np.newaxis], A_meas.shape[3], axis=3)

    # Set background of measured attenuations to zero
    A_meas_no_bkg = A_meas.copy()*rep_mask    
    
    # Full bi-tensor forward model values (one for each gradient direction) 
    A_bi = fwd_model(ss_bvecs, D_fit_2, f_fit_2)
        
    # Don't let attenuations stray outside bounds
    np.clip(A_bi, 0, 1, out=A_bi) 
        
    # Apply brain mask to results for each grad. dir.
    A_bi_no_bkg = A_bi * rep_mask

    # Euclidean distances between model and measured values (per voxel)
    A_bi_resid = np.linalg.norm(A_bi_no_bkg - A_meas_no_bkg, axis=3)    
    
    bi_resid_img = nib.Nifti1Image(A_bi_resid.astype(np.float32), nii.affine)
    sc_resid_img = nib.Nifti1Image(A_sc_resid.astype(np.float32), nii.affine)

    nib.save(bi_resid_img, cur_out_dir + 'bi_residual_img.nii.gz')
    nib.save(sc_resid_img, cur_out_dir + 'sc_residual_img.nii.gz')   
    
    # Concatenate lists and plot
    plt.plot(costs_1 + costs_2)
    plt.show()
    
    # Output stats
    
    # Bi-compartment model error
    bi_error = A_bi_resid.sum()
    print(bi_error)
    
    # Single compartment forward model values (one for each gradient direction) (f's == 1)
    A_sc_no_bkg = fwd_model(ss_bvecs, D0, np.ones_like(f0))*rep_mask
    np.clip(A_sc_no_bkg, 0, 1, out=A_sc_no_bkg) * rep_mask
    A_sc_resid = np.linalg.norm(A_sc_no_bkg - A_meas_no_bkg, axis=3)
    sc_error = A_sc_resid.sum()
    print(sc_error)
    
    # Total # of voxels (in single image) involved in fit
    print((A_bi_no_bkg[:,:,:,0] > 0).sum())
    
    # Voxels where single compartment fit is better
    print(((A_bi_resid - A_sc_resid) > 0).sum())
    
    # Voxels where bi-compartment fit is better
    print(((A_sc_resid - A_bi_resid) > 0).sum())
      
        
# Return to start directory
os.chdir(start_dir)
    
    

