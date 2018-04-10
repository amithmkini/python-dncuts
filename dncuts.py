"""
A_down = A;
SZ_down = SZ;
Bs = cell(N_DOWNSAMPLE,1);

for di = 1:N_DOWNSAMPLE
  
  % Create a binary array of the pixels that will remain after decimating
  
  % every other row and column
  [i,j] = ind2sub(SZ_down, 1:size(A_down,1));
  do_keep = (mod(i, DECIMATE) == 0) & (mod(j, DECIMATE) == 0);
  
  % Downsample the affinity matrix
  A_sub = A_down(:,do_keep)';
  
  % Normalize the downsampled affinity matrix
  d = (sum(A_sub,1) + eps);
  B = bsxfun(@rdivide, A_sub, d)';
  
  % "Square" the affinity matrix, while downsampling
  A_down = A_sub*B;

  SZ_down = floor(SZ_down / 2);
  
  % Hold onto the normalized affinity matrix for bookkeeping
  Bs{di} = B;  

end

% Get the eigenvectors of the Laplacian
%EV = ncuts(A_down, NVEC);
[EV, EVal] = ncuts(A_down, NVEC);

% Upsample the eigenvectors
for di = N_DOWNSAMPLE:-1:1
  EV = Bs{di} * EV;
end

% "Upsample" the eigenvalues (I'm not sure why this works, but it seems
% reasonable)
EVal = (2 ^ -N_DOWNSAMPLE) * EVal;

% whiten the eigenvectors, as they can get scaled weirdly during upsampling
EV = whiten(EV, 1, 0);
"""
import numpy as np


def dncuts(A, NVEC, N_DOWNSAMPLE, DECIMATE, SZ):
    # A = affinity matrix
    # NEVC = number of eigenvectors (set to 16?)
    # N_DOWNSAMPLE = number of downsampling operations (2 seems okay)
    # DECIMATE = amount of decimation for each downsampling operation (set to 2)
    # SZ = size of the image corresponding to A

    A_down = A
    SZ_down = SZ
    Bs = {}

    for di in range(N_DOWNSAMPLE):
        # i, j = np.ravel_multi_index(SZ_down, range(A_down.shape[0]))
        (j, i) = np.unravel_index(range(A_down.shape[0]), SZ_down)
        do_keep = (i%DECIMATE == 0) and (j%DECIMATE == 0)
        A_sub = np.transpose(A_down[:,do_keep])
        d = np.sum(A_sub, 0)