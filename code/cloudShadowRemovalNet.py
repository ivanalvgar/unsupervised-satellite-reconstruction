import torch.nn as nn
import torch.nn.functional as F
import torch


class DownBlock(nn.Module):
  """
  A downsampling block consisting of a convolutional layer followed by batch normalization and ReLU activation.

  This block reduces the spatial dimensions (height and width) of the input while increasing the depth (number of channels).

  Parameters
  ----------
  in_channels : int
      The number of input channels (depth) of the input tensor.
  out_channels : int
      The number of output channels (depth) after the convolution.
  kernel_size : int or tuple
      The size of the convolutional kernel.
  stride : int or tuple
      The stride of the convolution, which controls the downsampling rate.
  padding : int or tuple
      The padding added to the input tensor to ensure the correct output size.

  Attributes
  ----------
  conv : nn.Conv2d
      A 2D convolution layer that applies the kernel and performs downsampling.
  bn : nn.BatchNorm2d
      A 2D batch normalization layer that normalizes the output of the convolution to improve convergence.
  relu : nn.ReLU
      A ReLU activation function to introduce non-linearity.
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super().__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
      self.bn = nn.BatchNorm2d(out_channels)
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
      """
      Forward pass of the DownBlock.

      Applies the convolution, batch normalization, and ReLU activation to the input tensor.

      Parameters
      ----------
      x : torch.Tensor
          The input tensor with shape (batch_size, in_channels, height, width).

      Returns
      -------
      torch.Tensor
          The output tensor after passing through convolution, batch normalization, and ReLU activation.
      """
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x


class UpBlock(nn.Module):
  """
  An upsampling block consisting of a transposed convolution layer followed by batch normalization and ReLU activation.

  This block increases the spatial dimensions (height and width) of the input while decreasing the depth (number of channels).

  Parameters
  ----------
  in_channels : int
      The number of input channels (depth) of the input tensor.
  out_channels : int
      The number of output channels (depth) after the transposed convolution.
  kernel_size : int or tuple
      The size of the transposed convolution kernel.
  stride : int or tuple
      The stride of the transposed convolution, which controls the upsampling rate.
  padding : int or tuple
      The padding added to the input tensor to ensure the correct output size.

  Attributes
  ----------
  conv_transpose : nn.ConvTranspose2d
      A 2D transposed convolution layer (also known as a deconvolution) that applies the kernel and performs upsampling.
  bn : nn.BatchNorm2d
      A 2D batch normalization layer that normalizes the output of the transposed convolution to improve convergence.
  relu : nn.ReLU
      A ReLU activation function to introduce non-linearity.
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super().__init__()
      self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
      self.bn = nn.BatchNorm2d(out_channels)
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
      """
      Forward pass of the UpBlock.

      Applies the transposed convolution, batch normalization, and ReLU activation to the input tensor.

      Parameters
      ----------
      x : torch.Tensor
          The input tensor with shape (batch_size, in_channels, height, width).

      Returns
      -------
      torch.Tensor
          The output tensor after passing through transposed convolution, batch normalization, and ReLU activation.
      """
      x = self.conv_transpose(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class FinalBlock(nn.Module):
  """
  A final block consisting of a convolutional layer followed by a ReLU activation.

  This block is typically used to generate the final output tensor (e.g., in a segmentation task or output layer of a neural network).

  Parameters
  ----------
  in_channels : int
      The number of input channels (depth) of the input tensor.
  out_channels : int
      The number of output channels (depth) after the convolution.
  kernel_size : int or tuple
      The size of the convolutional kernel.
  stride : int or tuple
      The stride of the convolution, typically set to 1 in the final layer to preserve spatial dimensions.
  padding : int or tuple
      The padding added to the input tensor to ensure the correct output size.

  Attributes
  ----------
  conv : nn.Conv2d
      A 2D convolution layer to process the input tensor.
  relu : nn.ReLU
      A ReLU activation function to introduce non-linearity.
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super().__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
      """
      Forward pass of the FinalBlock.

      Applies the convolution and ReLU activation to the input tensor.

      Parameters
      ----------
      x : torch.Tensor
          The input tensor with shape (batch_size, in_channels, height, width).

      Returns
      -------
      torch.Tensor
          The output tensor after passing through convolution and ReLU activation.
      """
      x = self.conv(x)
      x = self.relu(x)
      return x

class CloudShadowRemovalNet(nn.Module):
  """
  A neural network model for cloud removal, utilizing an encoder-decoder architecture.

  The network consists of a series of downsampling (convolutional) blocks followed by upsampling (transposed convolution) blocks.
  The central_tensor method is used to extract the central portion of the feature maps from each downsampling block and concatenate them
  to the upsampling blocks for skip connections.

  Attributes
  ----------
  block1 to block15 : nn.Module
      Layers of the network, consisting of DownBlock, UpBlock, and FinalBlock.
  """

  def __init__(self):
      """
      Initializes the CloudShadowRemovalNet model with a series of DownBlock layers, UpBlock layers, and a FinalBlock.

      The downsampling blocks (block1 to block7) progressively reduce the spatial dimensions and increase the depth of the features.
      The upsampling blocks (block8 to block14) progressively increase the spatial dimensions while reducing the depth,
      concatenating the feature maps from corresponding downsampling layers for skip connections.
      The final block (block15) is a convolutional layer that outputs the final prediction.
      """
      super().__init__()
      # Downsampling blocks (Encoder)
      self.block1 = DownBlock(1, 128, 3, 2, 1)
      self.block2 = DownBlock(128, 256, 3, 2, 1)
      self.block3 = DownBlock(256, 512, 3, 2, 1)
      self.block4 = DownBlock(512, 1024, 3, 2, 1)
      self.block5 = DownBlock(1024, 2048, 3, 2, 1)
      self.block6 = DownBlock(2048, 2048, 3, 2, 1)
      self.block7 = DownBlock(2048, 2048, 3, 2, 1)

      # Upsampling blocks (Decoder)
      self.block8 = UpBlock(2048, 2048, 3, 2, 1)
      self.block9 = UpBlock(4096, 2048, 3, 2, 1)
      self.block10 = UpBlock(4096, 1024, 3, 2, 1)
      self.block11 = UpBlock(2048, 512, 3, 2, 1)
      self.block12 = UpBlock(1024, 256, 3, 2, 1)
      self.block13 = UpBlock(512, 128, 3, 2, 1)
      self.block14 = UpBlock(256, 64, 3, 2, 1)

      # Final convolutional block
      self.block15 = FinalBlock(64, 48, 3, 1, 1)

  def forward(self, x):
      """
      Forward pass of the CloudShadowRemovalNet.

      This method defines the forward propagation through the downsampling (encoder) and upsampling (decoder) blocks,
      utilizing skip connections between corresponding downsampling and upsampling layers.

      Parameters
      ----------
      x : torch.Tensor
          Input tensor with shape (batch_size, 1, height, width).

      Returns
      -------
      torch.Tensor
          The final output tensor with shape (batch_size, 48, height, width), which represents the cloud removal result.
      """
      # Downsampling through the encoder blocks
      output1 = self.block1(x)
      output2 = self.block2(output1)
      output3 = self.block3(output2)
      output4 = self.block4(output3)
      output5 = self.block5(output4)
      output6 = self.block6(output5)
      output7 = self.block7(output6)

      # Upsampling through the decoder blocks with skip connections
      output8 = self.block8(output7)
      output9 = self.block9(torch.cat((output8, self.central_tensor(output6, output8.shape[-1])), dim=1))
      output10 = self.block10(torch.cat((output9, self.central_tensor(output5, output9.shape[-1])), dim=1))
      output11 = self.block11(torch.cat((output10, self.central_tensor(output4, output10.shape[-1])), dim=1))
      output12 = self.block12(torch.cat((output11, self.central_tensor(output3, output11.shape[-1])), dim=1))
      output13 = self.block13(torch.cat((output12, self.central_tensor(output2, output12.shape[-1])), dim=1))
      output14 = self.block14(torch.cat((output13, self.central_tensor(output1, output13.shape[-1])), dim=1))

      # Final output layer
      final_output = self.block15(output14)
      return final_output

  def central_tensor(self, tensor, length):
      """
      Extracts the central portion of the tensor.

      This method extracts the central region of the input tensor to match the dimensions of the output tensor at each step of the network.

      Parameters
      ----------
      tensor : torch.Tensor
          The input tensor from which the central part will be extracted.
      length : int
          The length of the central portion to extract.

      Returns
      -------
      torch.Tensor
          The extracted central portion of the tensor with the specified length.
      """
      num_columns = tensor.shape[-1]
      start = (num_columns - length) // 2
      rows = torch.narrow(tensor, dim=2, start=start, length=length)
      central_tensor = torch.narrow(rows, dim=3, start=start, length=length)
      return central_tensor.clone()

class CustomLoss(nn.Module):
  """
  A custom loss function for image processing, combining Mean Squared Error (MSE) loss over masked image regions.

  This loss function applies the MSE loss to different image regions based on a given mask, which is used to focus on specific parts of the images.
  This function differs from 4.1 only by a scaling factor t, which does not affect the argument being optimized.

  Attributes
  ----------
  loss : nn.MSELoss
      A built-in loss function from PyTorch used to calculate the Mean Squared Error.
  """

  def __init__(self):
      """
      Initializes the CustomLoss function with the Mean Squared Error (MSE) loss.

      The MSE loss function is used to compute the squared difference between the predicted and true values,
      which helps in minimizing the error during training.

      This initialization sets up the loss function but does not require any additional parameters or layers.
      """
      super(CustomLoss, self).__init__()
      self.loss = nn.MSELoss()

  def forward(self, outputs, mask, image1, image2, image3, image4):
      """
      Forward pass of the CustomLoss function.

      This method calculates the total loss by applying the MSE loss function on masked regions of the output and the ground truth images.
      The mask is used to selectively include certain parts of the images in the loss calculation.

      Parameters
      ----------
      outputs : torch.Tensor
          The predicted tensor (batch_size, channels, height, width), which represents the model's output.

      mask : torch.Tensor
          A tensor of the same shape as the input and output images, with binary values where 1 represents the region to focus on and 0 otherwise.

      image1, image2, image3, image4 : torch.Tensor
          The ground truth images that the model is trying to predict, with the same shape as the output.

      Returns
      -------
      torch.Tensor
          The total loss computed by summing up the MSE losses for each masked image region. The output is a scalar tensor representing the total loss.
      """
      return (self.loss(mask[:12] * outputs[:12], mask[:12] * image1) +
              self.loss(mask[12:24] * outputs[12:24], mask[12:24] * image2) +
              self.loss(mask[24:36] * outputs[24:36], mask[24:36] * image3) +
              self.loss(mask[36:] * outputs[36:], mask[36:] * image4))
