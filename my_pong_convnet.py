""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses Gymnasium. """
import numpy as np
import pickle
import gymnasium

# hyperparameters
H = 128 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True

# model initialization
D = 40 * 40 * 64 # input dimensionality after flattening: 40x40x64 
if resume:
  model = pickle.load(open('save_conv.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory


def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

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

def convolve2d(images, filters): # Assumes same padding
    # Get dimensions of the image and filters
    batch_size, image_height, image_width, num_input_channels = images.shape
    filter_height, filter_width, num_input_channels, num_output_channels = filters.shape

    # Pad the image with zeros, same padding
    padding_height = filter_height // 2
    padding_width = filter_width // 2
    padded_images = np.pad(images, ((0,0), (padding_height, padding_height), (padding_width, padding_width), (0,0)), mode='constant')

    # Create an empty output array
    output = np.zeros((batch_size, image_height, image_width, num_output_channels))

    # Perform the convolution
    for b in range(batch_size):
        for i in range(image_height):
            for j in range(image_width):
                for k in range(num_output_channels):
                    output[b, i, j, k] = np.sum(padded_images[b, i:i+filter_height, j:j+filter_width, :] * filters[:, :, :, k])

    return output

def conv2D_forward(I, kernel_size, num_filters):  #2D conv using numpy
  filters = initialize_filters(kernel_size, num_filters)
  output = convolve2d(I, filters)
  return output

def forward_relu(input_tensor):
    # Apply ReLU operation element-wise
    output_tensor = np.maximum(0, input_tensor)
    
    return output_tensor

def forward_max_pooling_2d(input_tensor, pool_size=(2, 2)): #assume no stride
    # Get dimensions of the input tensor
    batch_size, input_height, input_width, num_channels = input_tensor.shape

    # Calculate output dimensions
    output_height = input_height // pool_size[0]
    output_width = input_width // pool_size[1]

    # Create an empty output tensor
    output_tensor = np.zeros((batch_size, output_height, output_width, num_channels))

    # Perform the max pooling
    for b in range(batch_size):
        for h in range(output_height):
            for w in range(output_width):
                for c in range(num_channels):
                    h_start = h * pool_size[0]
                    h_end = h_start + pool_size[0]
                    w_start = w * pool_size[1]
                    w_end = w_start + pool_size[1]

                    # Extract the corresponding patch from the input tensor
                    patch = input_tensor[b, h_start:h_end, w_start:w_end, c]

                    # Perform max pooling by taking the maximum value in the patch
                    output_tensor[b, h, w, c] = np.max(patch)

    return output_tensor
   

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  conv1 = conv2D_forward(x, kernel_size=(3,3), num_filters=32)
  relu1 = forward_relu(conv1)
  pool1 = forward_max_pooling_2d(relu1, pool_size=(2,2))

  conv2 = conv2D_forward(pool1, kernel_size=(3,3), num_filters=64)
  relu2 = forward_relu(conv2)
  pool2 = forward_max_pooling_2d(relu2, pool_size=(2,2))

  x_reshape = pool2.reshape(pool2.shape[0], -1) #flatten
  
  h = np.dot(model['W1'], x_reshape)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h, pool2, pool1 # return probability of taking action 2, and hidden state

def conv2D_backward(dL_dout, input_tensor, filters, output_shape):
    # Get dimensions
    batch_size, input_height, input_width, num_channels = input_tensor.shape
    filter_height, filter_width, num_input_channels, num_output_channels = filters.shape

    # Initialize gradients
    dL_dinput = np.zeros_like(input_tensor)
    dL_dfilters = np.zeros_like(filters)

    # Perform the backward pass
    for b in range(batch_size):
        for i in range(input_height):
            for j in range(input_width):
                for k in range(num_output_channels):
                    # Update gradient w.r.t. input tensor
                    dL_dinput[b, i:i+filter_height, j:j+filter_width, :] += dL_dout[b, i, j, k] * filters[:, :, :, k]

                    # Update gradient w.r.t. filters
                    dL_dfilters[:, :, :, k] += dL_dout[b, i, j, k] * input_tensor[b, i:i+filter_height, j:j+filter_width, :]

    # Remove padding from the gradients
    padding_height = filter_height // 2
    padding_width = filter_width // 2
    dL_dinput = dL_dinput[:, padding_height:-padding_height, padding_width:-padding_width, :]

    return dL_dinput, dL_dfilters


def backward_relu(dL_dout, input_tensor):
    # Compute element-wise derivative of ReLU
    dL_din = dL_dout * (input_tensor > 0)

    return dL_din


def backward_max_pooling_2d(dL_dout, input_tensor, pool_size=(2, 2)):
    # Get dimensions
    batch_size, input_height, input_width, num_channels = input_tensor.shape
    _, output_height, output_width, _ = dL_dout.shape

    # Initialize gradient
    dL_din = np.zeros_like(input_tensor)

    # Perform the backward pass
    for b in range(batch_size):
        for h in range(output_height):
            for w in range(output_width):
                for c in range(num_channels):
                    h_start = h * pool_size[0]
                    h_end = h_start + pool_size[0]
                    w_start = w * pool_size[1]
                    w_end = w_start + pool_size[1]

                    # Find the index of the maximum value in the corresponding input patch
                    max_index = np.argmax(input_tensor[b, h_start:h_end, w_start:w_end, c])

                    # Compute the gradient only at the maximum value
                    dL_din[b, h_start:h_end, w_start:w_end, c].flat[max_index] = dL_dout[b, h, w, c]

    return dL_din


def conv2D_backward_pass(dL_dout, input_tensor, filters, output_shape):
    # Backpropagate through the layers
    dL_din_relu = backward_max_pooling_2d(dL_dout, input_tensor, pool_size=(2, 2))
    dL_din = backward_relu(dL_din_relu, input_tensor)
    dL_din, dL_dfilters = conv2D_backward(dL_din, input_tensor, filters, output_shape)

    return dL_din, dL_dfilters

def policy_backward(eph, epdlogp, eppool2, eppool1):
  """ backward pass of dense layers. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backprop relu
  dW1 = np.dot(dh.T, epx)
  dflatten = np.dot(model['W1'].T, epdlogp)
  dP2 = dflatten.reshape(eppool2[0].shape)
  
  return {'W1':dW1, 'W2':dW2, 'K2': dK2, 'K1': dK1}, 

render_mode = 'human' if render else None

env = gymnasium.make("ALE/Pong-v5", render_mode = render_mode)
observation, info = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs, pool2s, pool1s = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h, pool2, pool1 = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, truncated, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    eppool1 = np.vstack(pool1s)
    eppool2 = np.vstack(pool2s)
    xs,hs,dlogps,drs,pool1s,pool2s = [],[],[],[],[],[]# reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp, eppool2, eppool1)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print(f"resetting env. episode reward total was {reward_sum}. running mean: {running_reward}")
    if episode_number % 100 == 0: pickle.dump(model, open('save_conv.p', 'wb'))
    reward_sum = 0
    observation, info = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print('ep %d: game finished, reward: %f%s' % (episode_number, reward, '' if reward == -1 else ' !!!!!!!!'))


