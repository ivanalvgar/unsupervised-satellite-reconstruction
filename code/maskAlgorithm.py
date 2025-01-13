import torch
import numpy as np

def fast_soft_thresholding(A,b):
  """
  Perform fast soft thresholding.

  Parameters
  ----------
  A : `torch.Tensor`
      Input tensor.
  b : float
      Threshold value.

  Returns
  -------
  FST : `torch.Tensor`
      Result after applying soft thresholding.
  """
  FST = torch.sign(A)*torch.maximum(torch.abs(A)-b,torch.tensor(0))
  return FST

def psf2otf(psf, shape):
  """
  Convert point-spread function to optical transfer function.

  Parameters
  ----------
  psf : `numpy.ndarray`
      PSF array.
  shape : int
      Output shape of the OTF array.

  Returns
  -------
  otf : `numpy.ndarray`
      OTF array.
  """
  if np.all(psf == 0):
      return np.zeros_like(psf)
  inshape = psf.shape
  psf = zero_pad(psf, shape, position='corner')
  for axis, axis_size in enumerate(inshape):
      psf = np.roll(psf, -int(axis_size / 2), axis=axis)
  otf = np.fft.fft2(psf)
  n_ops = np.sum(psf.size * np.log2(psf.shape))
  otf = np.real_if_close(otf, tol=n_ops)
  return otf

def zero_pad(image, shape, position='corner'):
  """
  Extend image to a certain size with zeros.

  Parameters
  ----------
  image : `numpy.ndarray`
      Input image.
  shape : tuple of int
      Desired output shape of the image.
  position : str, optional
      The position of the input image in the output:
          - 'corner': Top-left corner (default).
          - 'center': Centered.

  Returns
  -------
  padded_img : `numpy.ndarray`
      Zero-padded image.
  """
  shape = np.asarray(shape, dtype=int)
  imshape = np.asarray(image.shape, dtype=int)
  if np.alltrue(imshape == shape):
      return image
  if np.any(shape <= 0):
      raise ValueError("ZERO_PAD: null or negative shape given")
  dshape = shape - imshape
  if np.any(dshape < 0):
      raise ValueError("ZERO_PAD: target size smaller than source one")
  pad_img = np.zeros(shape, dtype=image.dtype)
  idx, idy = np.indices(imshape)
  if position == 'center':
      if np.any(dshape % 2 != 0):
          raise ValueError("ZERO_PAD: source and target shapes "
                            "have different parity.")
      offx, offy = dshape // 2
  else:
      offx, offy = (0, 0)
  pad_img[idx + offx, idy + offy] = image
  return pad_img

def diffX(S):
  """
  Compute finite differences along the x-axis.

  Parameters
  ----------
  S : `torch.Tensor`
      Input tensor of shape (b, m, n).

  Returns
  -------
  aux : `torch.Tensor`
      Tensor with computed finite differences.
  """
  b, m, n = S.shape
  aux = torch.zeros(S.shape)
  dfx = torch.diff(S, dim=1)
  aux[:,0:m-1,:] = dfx
  aux[:,-1,:] = S[:,0,:] - S[:,-1,:]
  return aux

def diffY(S):
  """
  Compute finite differences along the y-axis.

  Parameters
  ----------
  S : `torch.Tensor`
      Input tensor of shape (b, m, n).

  Returns
  -------
  aux : `torch.Tensor`
      Tensor with computed finite differences.
  """
  b, m, n = S.shape
  aux = torch.zeros(S.shape)
  dfy = torch.diff(S, dim=2)
  aux[:,:,0:n-1] = dfy
  aux[:,:,-1] = S[:,:,0] - S[:,:,-1]
  return aux

def diffZ(S):
  """
  Compute finite differences along the z-axis.

  Parameters
  ----------
  S : `torch.Tensor`
      Input tensor of shape (b, m, n).

  Returns
  -------
  aux : `torch.Tensor`
      Tensor with computed finite differences.
  """
  b, m, n = S.shape
  aux = torch.zeros(S.shape)
  dfz = torch.diff(S, dim=0)
  aux[0:b-1,:,:] = dfz
  aux[-1,:,:] = S[0,:,:] - S[-1,:,:]
  return aux

def diffT3(B):
  """
  Compute transpose of finite differences for a 3D tensor.

  Parameters
  ----------
  B : `torch.Tensor`
      Input tensor of shape (3, b, m, n), representing the 3D gradients.

  Returns
  -------
  diffT : `torch.Tensor`
      Transpose of finite differences.
  """
  dif, b, m, n = B.shape
  Bx = B[0,:,:,:]
  By = B[1,:,:,:]
  Bz = B[2,:,:,:]

  dfxT = torch.zeros(b, m, n)
  dfyT = torch.zeros(b, m, n)
  dfzT = torch.zeros(b, m, n)

  dfx = torch.diff(Bx, dim=1)
  dfy = torch.diff(By, dim=2)
  dfz = torch.diff(Bz, dim=0)

  dfxT[:,0,:] = Bx[:,-1,:] - Bx[:,0,:]
  dfxT[:,1:m,:] = -dfx
  dfyT[:,:,0] = By[:,:,-1] - By[:,:,0]
  dfyT[:,:,1:n] = -dfy
  dfzT[0,:,:] = Bz[-1,:,:] - Bz[0,:,:]
  dfzT[1:b,:,:] = -dfz
  return dfxT + dfyT + dfzT

def svd(X, lambda1, beta):
  """
  Perform singular value decomposition with thresholding.

  Parameters
  ----------
  X : `torch.Tensor`
      Input tensor of shape (t, b, m, n).
  lambda1 : float
      Regularization parameter.
  beta : float
      Scaling parameter.

  Returns
  -------
  new_X : `torch.Tensor`
      Tensor after applying truncated SVD and thresholding.
  """
  t, b, m, n = X.shape
  new_X = torch.zeros(t,b,m,n)
  array = torch.reshape(X, (t*b,m,n))
  array = np.array(array)
  columns = []
  for matrix in array:
      column = matrix.reshape((-1, 1),order='F')
      columns.append(column)
  matrix = torch.tensor(np.concatenate(columns, axis=1))
  U, sigma, V = torch.svd(matrix)
  U = np.array(U)
  sigma = np.array(sigma)
  V = np.array(V)
  svp = np.sum(sigma > lambda1/beta)
  sigma_truncated = np.diag(sigma[:svp] - lambda1/beta)
  U_svp = U[:, :svp]
  V_svp = V[:, :svp]
  aux = np.dot(np.dot(U_svp, sigma_truncated), V_svp.T)
  matrixes = []
  for columna in aux.T:
      matrix = columna.reshape((m, -n),order='F')
      matrixes.append(matrix)
  for i in range(b*t):
    new_X[i//b,i % b,:,:] = torch.tensor(matrixes[i])
  return new_X

def laplace(B):
  """
  Compute the Laplacian of a 3D tensor using finite differences.

  Parameters
  ----------
  B : torch.Tensor
      Input 3D tensor of shape (m, n, b) where m, n, and b are the
      dimensions of the tensor.

  Returns
  -------
  aux : torch.Tensor
      Output tensor of the same shape as B, containing the Laplacian
      values computed using discrete approximations.
  """
  m, n, b = B.shape
  aux = torch.zeros(m,n,b)
  aux[1:-1, 1:-1, 1:-1] = (B[:-2, 1:-1, 1:-1] + B[2:, 1:-1, 1:-1] +
                              B[1:-1, :-2, 1:-1] + B[1:-1, 2:, 1:-1] +
                              B[1:-1, 1:-1, :-2] + B[1:-1, 1:-1, 2:])

  aux[1:-1, 1:-1, 1:-1] = -8 * B[1:-1, 1:-1, 1:-1] + aux[1:-1, 1:-1, 1:-1]
  return aux

def ADMM(Y, lambda1, lambda2, lambda3, beta, epsilon, kmax):
  """
  Alternating Direction Method of Multipliers (ADMM) for tensor decomposition.

  Parameters
  ----------
  Y : `torch.Tensor`
      Input tensor of shape (t, b, m, n).
  lambda1 : float
      Regularization parameter for the low-rank component.
  lambda2 : float
      Regularization parameter for sparsity.
  lambda3 : float
      Regularization parameter for the gradient.
  beta : float
      ADMM penalty parameter.
  epsilon : float
      Convergence tolerance.
  kmax : int
      Maximum number of iterations.

  Returns
  -------
  X : `torch.Tensor`
      Low-rank component of the decomposition.
  S : `torch.Tensor`
      Sparse component of the decomposition.
  N : `torch.Tensor`
      Noise component of the decomposition.
  """
  t, b, m, n = Y.shape
  X = Y.clone()
  S = torch.zeros((t, b, m, n))
  N = torch.zeros((t, b, m, n))
  R = torch.zeros((t, b, m, n))
  W1 = torch.zeros((t, b, m, n))
  W2 = torch.zeros((t, b, m, n))
  Q = torch.zeros((3, t, b, m, n))
  A = torch.zeros((3, t, b, m, n))

  for k in range(kmax):
    R = fast_soft_thresholding(S-W2/beta,lambda2/beta)

    for i in range(0,t):
      Q[0, i] = fast_soft_thresholding(diffX(S[i])-A[0,i]/beta,lambda3/beta)
      Q[1, i] = fast_soft_thresholding(diffY(S[i])-A[1,i]/beta,lambda3/beta)
      Q[2, i] = fast_soft_thresholding(diffZ(S[i])-A[2,i]/beta,lambda3/beta)

    Xb = X.clone()
    X = svd(Y - S - N + W1/beta, lambda1, beta)

    for i in range(0,t):
      for j in range(10):
        Sa = S[i,:,:,:].clone()
        DSa = laplace(Sa)
        S[i,:,:,:] = (Sa + (0.005)*(DSa + R[i] + W2[i]/beta + Y[i] - X[i] - N[i] + W1[i]/beta + diffT3(Q[:,i,:,:,:]+A[:,i,:,:,:]/beta)))

    N = (Y - X - S + W1/beta)*beta/(beta+1)

    W1 = W1 + beta*(Y - S - X - N)
    W2 = W2 + beta*(R - S)
    for i in range(0,t):
      A[0,i] = A[0,i] + beta*(Q[0,i] - diffX(S[i]))
      A[1,i] = A[1,i] + beta*(Q[1,i] - diffY(S[i]))
      A[2,i] = A[2,i] + beta*(Q[2,i] - diffZ(S[i]))

    if (torch.norm(X - Xb)/torch.norm(X) <= epsilon):
      print("Min Error, Iteration: " + str(k))
      break
    elif (k == kmax):
      print("Max Iterations")
      break
  return X, S, N

def fast_soft_thresholding(A,b):
  """
  Perform fast soft thresholding.

  Parameters
  ----------
  A : `torch.Tensor`
      Input tensor.
  b : float
      Threshold value.

  Returns
  -------
  FST : `torch.Tensor`
      Result after applying soft thresholding.
  """
  FST = torch.sign(A)*torch.maximum(torch.abs(A)-b,torch.tensor(0))
  return FST

def psf2otf(psf, shape):
  """
  Convert point-spread function to optical transfer function.

  Parameters
  ----------
  psf : `numpy.ndarray`
      PSF array.
  shape : int
      Output shape of the OTF array.

  Returns
  -------
  otf : `numpy.ndarray`
      OTF array.
  """
  if np.all(psf == 0):
      return np.zeros_like(psf)
  inshape = psf.shape
  psf = zero_pad(psf, shape, position='corner')
  for axis, axis_size in enumerate(inshape):
      psf = np.roll(psf, -int(axis_size / 2), axis=axis)
  otf = np.fft.fft2(psf)
  n_ops = np.sum(psf.size * np.log2(psf.shape))
  otf = np.real_if_close(otf, tol=n_ops)
  return otf

def zero_pad(image, shape, position='corner'):
  """
  Extend image to a certain size with zeros.

  Parameters
  ----------
  image : `numpy.ndarray`
      Input image.
  shape : tuple of int
      Desired output shape of the image.
  position : str, optional
      The position of the input image in the output:
          - 'corner': Top-left corner (default).
          - 'center': Centered.

  Returns
  -------
  padded_img : `numpy.ndarray`
      Zero-padded image.
  """
  shape = np.asarray(shape, dtype=int)
  imshape = np.asarray(image.shape, dtype=int)
  if np.alltrue(imshape == shape):
      return image
  if np.any(shape <= 0):
      raise ValueError("ZERO_PAD: null or negative shape given")
  dshape = shape - imshape
  if np.any(dshape < 0):
      raise ValueError("ZERO_PAD: target size smaller than source one")
  pad_img = np.zeros(shape, dtype=image.dtype)
  idx, idy = np.indices(imshape)
  if position == 'center':
      if np.any(dshape % 2 != 0):
          raise ValueError("ZERO_PAD: source and target shapes "
                            "have different parity.")
      offx, offy = dshape // 2
  else:
      offx, offy = (0, 0)
  pad_img[idx + offx, idy + offy] = image
  return pad_img

def diffX(S):
  """
  Compute finite differences along the x-axis.

  Parameters
  ----------
  S : `torch.Tensor`
      Input tensor of shape (b, m, n).

  Returns
  -------
  aux : `torch.Tensor`
      Tensor with computed finite differences.
  """
  b, m, n = S.shape
  aux = torch.zeros(S.shape)
  dfx = torch.diff(S, dim=1)
  aux[:,0:m-1,:] = dfx
  aux[:,-1,:] = S[:,0,:] - S[:,-1,:]
  return aux

def diffY(S):
  """
  Compute finite differences along the y-axis.

  Parameters
  ----------
  S : `torch.Tensor`
      Input tensor of shape (b, m, n).

  Returns
  -------
  aux : `torch.Tensor`
      Tensor with computed finite differences.
  """
  b, m, n = S.shape
  aux = torch.zeros(S.shape)
  dfy = torch.diff(S, dim=2)
  aux[:,:,0:n-1] = dfy
  aux[:,:,-1] = S[:,:,0] - S[:,:,-1]
  return aux

def diffZ(S):
  """
  Compute finite differences along the z-axis.

  Parameters
  ----------
  S : `torch.Tensor`
      Input tensor of shape (b, m, n).

  Returns
  -------
  aux : `torch.Tensor`
      Tensor with computed finite differences.
  """
  b, m, n = S.shape
  aux = torch.zeros(S.shape)
  dfz = torch.diff(S, dim=0)
  aux[0:b-1,:,:] = dfz
  aux[-1,:,:] = S[0,:,:] - S[-1,:,:]
  return aux

def diffT3(B):
  """
  Compute transpose of finite differences for a 3D tensor.

  Parameters
  ----------
  B : `torch.Tensor`
      Input tensor of shape (3, b, m, n), representing the 3D gradients.

  Returns
  -------
  diffT : `torch.Tensor`
      Transpose of finite differences.
  """
  dif, b, m, n = B.shape
  Bx = B[0,:,:,:]
  By = B[1,:,:,:]
  Bz = B[2,:,:,:]

  dfxT = torch.zeros(b, m, n)
  dfyT = torch.zeros(b, m, n)
  dfzT = torch.zeros(b, m, n)

  dfx = torch.diff(Bx, dim=1)
  dfy = torch.diff(By, dim=2)
  dfz = torch.diff(Bz, dim=0)

  dfxT[:,0,:] = Bx[:,-1,:] - Bx[:,0,:]
  dfxT[:,1:m,:] = -dfx
  dfyT[:,:,0] = By[:,:,-1] - By[:,:,0]
  dfyT[:,:,1:n] = -dfy
  dfzT[0,:,:] = Bz[-1,:,:] - Bz[0,:,:]
  dfzT[1:b,:,:] = -dfz
  return dfxT + dfyT + dfzT

def svd(X, lambda1, beta):
  """
  Perform singular value decomposition with thresholding.

  Parameters
  ----------
  X : `torch.Tensor`
      Input tensor of shape (t, b, m, n).
  lambda1 : float
      Regularization parameter.
  beta : float
      Scaling parameter.

  Returns
  -------
  new_X : `torch.Tensor`
      Tensor after applying truncated SVD and thresholding.
  """
  t, b, m, n = X.shape
  new_X = torch.zeros(t,b,m,n)
  array = torch.reshape(X, (t*b,m,n))
  array = np.array(array)
  columns = []
  for matrix in array:
      column = matrix.reshape((-1, 1),order='F')
      columns.append(column)
  matrix = torch.tensor(np.concatenate(columns, axis=1))
  U, sigma, V = torch.svd(matrix)
  U = np.array(U)
  sigma = np.array(sigma)
  V = np.array(V)
  svp = np.sum(sigma > lambda1/beta)
  sigma_truncated = np.diag(sigma[:svp] - lambda1/beta)
  U_svp = U[:, :svp]
  V_svp = V[:, :svp]
  aux = np.dot(np.dot(U_svp, sigma_truncated), V_svp.T)
  matrixes = []
  for columna in aux.T:
      matrix = columna.reshape((m, -n),order='F')
      matrixes.append(matrix)
  for i in range(b*t):
    new_X[i//b,i % b,:,:] = torch.tensor(matrixes[i])
  return new_X

def laplace(B):
  """
  Compute the Laplacian of a 3D tensor using finite differences.

  Parameters
  ----------
  B : torch.Tensor
      Input 3D tensor of shape (m, n, b) where m, n, and b are the
      dimensions of the tensor.

  Returns
  -------
  aux : torch.Tensor
      Output tensor of the same shape as B, containing the Laplacian
      values computed using discrete approximations.
  """
  m, n, b = B.shape
  aux = torch.zeros(m,n,b)
  aux[1:-1, 1:-1, 1:-1] = (B[:-2, 1:-1, 1:-1] + B[2:, 1:-1, 1:-1] +
                              B[1:-1, :-2, 1:-1] + B[1:-1, 2:, 1:-1] +
                              B[1:-1, 1:-1, :-2] + B[1:-1, 1:-1, 2:])

  aux[1:-1, 1:-1, 1:-1] = -8 * B[1:-1, 1:-1, 1:-1] + aux[1:-1, 1:-1, 1:-1]
  return aux

def ADMM(Y, lambda1, lambda2, lambda3, beta, epsilon, kmax):
  """
  Alternating Direction Method of Multipliers (ADMM) for tensor decomposition.

  Parameters
  ----------
  Y : `torch.Tensor`
      Input tensor of shape (t, b, m, n).
  lambda1 : float
      Regularization parameter for the low-rank component.
  lambda2 : float
      Regularization parameter for sparsity.
  lambda3 : float
      Regularization parameter for the gradient.
  beta : float
      ADMM penalty parameter.
  epsilon : float
      Convergence tolerance.
  kmax : int
      Maximum number of iterations.

  Returns
  -------
  X : `torch.Tensor`
      Low-rank component of the decomposition.
  S : `torch.Tensor`
      Sparse component of the decomposition.
  N : `torch.Tensor`
      Noise component of the decomposition.
  """
  t, b, m, n = Y.shape
  X = Y.clone()
  S = torch.zeros((t, b, m, n))
  N = torch.zeros((t, b, m, n))
  R = torch.zeros((t, b, m, n))
  W1 = torch.zeros((t, b, m, n))
  W2 = torch.zeros((t, b, m, n))
  Q = torch.zeros((3, t, b, m, n))
  A = torch.zeros((3, t, b, m, n))

  for k in range(kmax):
    R = fast_soft_thresholding(S-W2/beta,lambda2/beta)

    for i in range(0,t):
      Q[0, i] = fast_soft_thresholding(diffX(S[i])-A[0,i]/beta,lambda3/beta)
      Q[1, i] = fast_soft_thresholding(diffY(S[i])-A[1,i]/beta,lambda3/beta)
      Q[2, i] = fast_soft_thresholding(diffZ(S[i])-A[2,i]/beta,lambda3/beta)

    Xb = X.clone()
    X = svd(Y - S - N + W1/beta, lambda1, beta)

    for i in range(0,t):
      for j in range(10):
        Sa = S[i,:,:,:].clone()
        DSa = laplace(Sa)
        S[i,:,:,:] = (Sa + (0.005)*(DSa + R[i] + W2[i]/beta + Y[i] - X[i] - N[i] + W1[i]/beta + diffT3(Q[:,i,:,:,:]+A[:,i,:,:,:]/beta)))

    N = (Y - X - S + W1/beta)*beta/(beta+1)

    W1 = W1 + beta*(Y - S - X - N)
    W2 = W2 + beta*(R - S)
    for i in range(0,t):
      A[0,i] = A[0,i] + beta*(Q[0,i] - diffX(S[i]))
      A[1,i] = A[1,i] + beta*(Q[1,i] - diffY(S[i]))
      A[2,i] = A[2,i] + beta*(Q[2,i] - diffZ(S[i]))

    if (torch.norm(X - Xb)/torch.norm(X) <= epsilon):
      print("Min Error, Iteration: " + str(k))
      break
    elif (k == kmax):
      print("Max Iterations")
      break
  return X, S, N

def detection(S, tau1, tau2):
  """
  Detect anomalies in a 4D tensor based on spectral mean thresholds.

  Parameters
  ----------
  S : torch.Tensor
      Input 4D tensor of shape (t, b, m, n), where:
          - t : Number of temporal frames
          - b : Spectral bands
          - m : Spatial height
          - n : Spatial width
  tau1 : float
      Positive threshold for detecting anomalies.
  tau2 : float
      Negative threshold for detecting anomalies.

  Returns
  -------
  mask : torch.Tensor
      3D binary mask of shape (t, m, n) indicating normal regions (1)
      and detected anomalies (0).
  Sn : torch.Tensor
      4D tensor of the same shape as S, where spectral components
      with a mean greater than `tau1` are isolated.
  Ss : torch.Tensor
      4D tensor of the same shape as S, where spectral components
      with a mean less than `-tau2` are isolated.
  """
  t, b, m, n = S.shape
  mean_S = torch.mean(S, dim=1)
  mask = torch.ones(t, m, n)
  Sn = torch.zeros_like(S)
  Ss = torch.zeros_like(S)
  mask[(mean_S > tau1) | (mean_S < -tau2)] = 0
  Sn[(mean_S > tau1).unsqueeze(1).expand_as(S)] = S[(mean_S > tau1).unsqueeze(1).expand_as(S)]
  Ss[(mean_S < -tau2).unsqueeze(1).expand_as(S)] = S[(mean_S < -tau2).unsqueeze(1).expand_as(S)]
  return mask, Sn, Ss