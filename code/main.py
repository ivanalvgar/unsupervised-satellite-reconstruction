import torch
import numpy as np
import matplotlib.pyplot as plt
from cloudShadowRemovalNet import CloudShadowRemovalNet, CustomLoss
from maskAlgorithm import ADMM, detection
from download import data

def cloud_shadow_detection_removal(I, lambda1, lambda2, lambda3, beta, epsilon, kmax, tau1, tau2, kmax2, stop_cond):
  """
  Perform cloud and shadow detection and removal on an input image using the ADMM algorithm
  and a neural network model (CloudShadowRemovalNet).

  The function first applies the ADMM algorithm to decompose the input image into components:
  X (original), S (shadow), and N (noise). Then, the detection function is used to detect
  clouds and shadows, generating a mask. The neural network is trained to remove clouds and shadows
  from the input image using this mask. The process iterates until a stopping criterion is met.

  Parameters
  ----------
  I : `torch.Tensor`
      A tensor of shape (4, 48, 300, 300) representing the input image (4 channels, 48 frames, 300x300 pixels).

  lambda1 : float
      Regularization parameter for the ADMM algorithm.

  lambda2 : float
      Regularization parameter for the ADMM algorithm.

  lambda3 : float
      Regularization parameter for the ADMM algorithm.

  beta : float
      The penalty parameter used in the ADMM algorithm.

  epsilon : float
      Convergence tolerance for the ADMM algorithm.

  kmax : int
      Maximum number of iterations for the ADMM algorithm.

  tau1 : float
      Threshold value for cloud detection.

  tau2 : float
      Threshold value for shadow detection.

  kmax2 : int
      Maximum number of epochs for training the neural network.

  stop : float
      Threshold for the stop condition based on the loss between the model output and the mask.

  Returns
  -------
  outputs : `torch.Tensor`
      The output tensor of shape (4, 12, 300, 300) after cloud and shadow removal.
  """
  #Step 1: Cloud and shadow mask
  X, S, N = ADMM(I, lambda1, lambda2, lambda3, beta, epsilon, kmax)

  mask, _, _ = detection(S, tau1, tau2)

  # Step 2: Convolutional Network
  width = I.shape[2]
  height = I.shape[3]
  net = CloudShadowRemovalNet()
  criterion = CustomLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.3)
  inputs = torch.randn([1, 1, width, height]) + torch.randn([1, 1, width, height]) * 0.1 + 0
  mask1 = mask[0].repeat(12, 1, 1).clone()
  mask2 = mask[1].repeat(12, 1, 1).clone()
  mask3 = mask[2].repeat(12, 1, 1).clone()
  mask4 = mask[3].repeat(12, 1, 1).clone()
  mask_loss = torch.cat((mask1, mask2, mask3, mask4), dim=0)
  for epoch in range(kmax2):
      outputs = net(inputs)
      loss = criterion(outputs.squeeze(0), mask_loss, I[0], I[1], I[2], I[3])
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      num = torch.norm((outputs.squeeze(0)[:12] - I[0] * mask_loss[0]), p=2) + \
            torch.norm((outputs.squeeze(0)[12:24] - I[1] * mask_loss[1]), p=2) + \
            torch.norm((outputs.squeeze(0)[24:36] - I[2] * mask_loss[2]), p=2) + \
            torch.norm((outputs.squeeze(0)[36:] - I[3] * mask_loss[3]), p=2)
      den = torch.sum(mask_loss)
      if den == 0:
        raise ValueError("The denominator is zero; please check the generated masks.")
      stop = num / den
      if epoch % 100 == 0:
          fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
          ax[0].imshow(np.dstack((
              outputs[0, 6, :, :].detach().numpy(),
              outputs[0, 2, :, :].detach().numpy(),
              outputs[0, 1, :, :].detach().numpy()
          )))
          ax[1].imshow(np.dstack((
              outputs[0, 18, :, :].detach().numpy(),
              outputs[0, 14, :, :].detach().numpy(),
              outputs[0, 13, :, :].detach().numpy()
          )))
          ax[2].imshow(np.dstack((
              outputs[0, 30, :, :].detach().numpy(),
              outputs[0, 26, :, :].detach().numpy(),
              outputs[0, 25, :, :].detach().numpy()
          )))
          ax[3].imshow(np.dstack((
              outputs[0, 42, :, :].detach().numpy(),
              outputs[0, 38, :, :].detach().numpy(),
              outputs[0, 37, :, :].detach().numpy()
          )))
          plt.show()
          print(f"Epoch {epoch}, stop criteria: {stop}")
      if stop < stop_cond:
          break
  print(f"Finished {epoch}, stop criteria: {stop}")
  new_I  = torch.empty_like(I)
  new_I[0] = outputs.squeeze(0)[:12]
  new_I[1] = outputs.squeeze(0)[12:24]
  new_I[2] = outputs.squeeze(0)[24:36]
  new_I[3] = outputs.squeeze(0)[36:]
  return new_I


if __name__ == "__main__":
    I = data([("2024-04-03", "2024-04-04"), ("2024-03-31", "2024-04-01"), ("2023-08-14", "2023-08-15"),
              ("2023-11-05", "2023-11-06")], (-5.663452, 43.443980, -5.545057, 43.537598), (385, 385))
    new_I = cloud_shadow_detection_removal(I, 6, 0.004, 0.005, 0.01, 0.0005, 15, 0.12, 0.02, 15000, 0.0001)
