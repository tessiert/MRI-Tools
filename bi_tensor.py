#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the bi-tensor gradient descent free water DTI analysis algorithm
as described in the paper "Free water elimination and mapping from diffusion MRI": 
Ofer Pasternak, Nir Sochen, Yaniv Gur, Nathan Intrator, Yaniv Assaf, 
First published: 21 July 2009 https://doi.org/10.1002/mrm.22055
"""

import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti



# Helper function to compute products of the form v_T*M*v, where v is a stack of vectors and M is a stack of matrices
def vec_mat_vec_prod(v, M):
    
    # Product M*v (i, j - dimensions of M, k - # of gradient directions => result is one 3-vec for each grad. dir.)
    temp = np.einsum('...ij,kj->...ik', M, v)
    
    # Product v_T*M*v (result is one value for each grad. dir.)
    return np.einsum('...ki,...ik->...k', v, temp)



# Bi-tensor DTI model forward function
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
        
        # Don't let attenuations stray outside bounds
        np.clip(A_bi, 0, 1, out=A_bi) 
        np.clip(A_tissue, 0, 1, out=A_tissue)
        
        # Apply brain mask to results for each grad. dir.
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
        print((A_diff + beltrami).sum())
        
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



def main(dti_file, bvals_file, bvecs_file, b_ss=1000):
    
    # Load the image data
    nii = nib.load(dti_file)
    img_data = nii.get_data()
    
    # Read in the b-shell values and gradient directions
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    
    # Boolean array to identify entries with either b = 0 or b = b_ss
    bvals_eq_0_b_ss = (bvals == 0) | (bvals == b_ss)

    # Extract info needed to run single-compartment dti model
    dti_bvals = bvals[bvals_eq_0_b_ss].copy()
    dti_bvecs = bvecs[bvals_eq_0_b_ss].copy()
    dti_img_data = img_data[:,:,:,bvals_eq_0_b_ss].copy()
    
    # Compute gradient table
    grad_table = gradient_table(dti_bvals, dti_bvecs)
    
    # Extract brain so we don't fit the background
    brain_img_data, brain_mask = median_otsu(dti_img_data, 2, 1) 

    # Run the dti model and fit it to the brain extracted image data
    ten_model = dti.TensorModel(grad_table)
    ten_fit = ten_model.fit(brain_img_data)
    
    
    
main()