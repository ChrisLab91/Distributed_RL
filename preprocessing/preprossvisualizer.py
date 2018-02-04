from grayscale import Grayscale
from image_resize import ImageResize
from sequence import Sequence
from crop import Crop
import numpy as np
import scipy.misc as smp
import scipy

import gym

### file for testing stuff...###

env = gym.make('SpaceInvaders-v0')
observation=env.reset()
for i_episode in range(30):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def zeroandone(image, threshold):
    image = (image.astype(np.float32) // threshold)*threshold
    return image

grayscale=Grayscale()
imageresize=ImageResize(84, 84)
sequence=Sequence(4)

#observation=sequence.process(observation)
#observation=imageresize.process(observation)
#observation=crop(observation,14,77,11,75)
#observation=crop(observation,36,190,20,144)
#imageresize=ImageResize(42,42)
#observation=grayscale.process(observation)
#observation=zeroandone(observation)

a=observation[30]
print(observation)
print(len(observation))
print(a)
print(len(a))
print(a[0])


#imageresize=ImageResize(336,336)
#observation=rgb2gray(observation)
#observation=zeroandone(observation, 105)
#observation=imageresize.process(observation)
#observation = scipy.misc.imresize(arr=observation.astype(np.uint8), size=(168, 168))
print(observation)


img = smp.toimage( observation )
smp.imsave('ong.png', img)
img.show()
