import numpy as np

def rgb_to_gray(rgb_image):
    # Extract the RGB channels
    red = rgb_image[:, :, 0]
    green = rgb_image[:, :, 1]
    blue = rgb_image[:, :, 2]
    
    # Compute the grayscale value
    gray_value = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    
    # Convert to uint8 data type (0-255)
    gray_image = gray_value.astype(np.uint8)
    
    return gray_image

def prepro(I):
  """ crop 210x160x3 uint8 frame into (160x160x3) and convert into grayscale (160*160)"""
  I = I[35:195] # crop
  I = rgb_to_gray(I)
  return I

def initialize_filters(filter_shape, num_filters):
    # Random initialization from a Gaussian distribution
    filters = np.random.normal(loc=0, scale=0.1, size=(filter_shape[0], filter_shape[1], num_filters))
    
    return filters

def convolve2d(image, filters): # Assumes same padding
    # Get dimensions of the image and filters
    image_height, image_width = image.shape
    filter_height, filter_width, num_filters = filters.shape

    # Pad the image with zeros, same padding
    padding_height = filter_height // 2
    padding_width = filter_width // 2
    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

    # Create an empty output array
    output = np.zeros((image_height, image_width, num_filters))

    # Perform the convolution
    for i in range(image_height):
        for j in range(image_width):
            for k in range(num_filters):
                output[i, j, k] = np.sum(padded_image[i:i+filter_height, j:j+filter_width] * filters[:, :, k])

    return output

def conv2D_forward(I, kernel_size, num_filters):  #2D conv using numpy
  filters = initialize_filters(kernel_size, num_filters)
  output = convolve2d(I, filters)
  return output


#lets try backprop

