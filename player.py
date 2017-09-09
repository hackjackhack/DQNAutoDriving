#!/usr/bin/python

import argparse
import random
import socket, sys
import struct
import time
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D

# Configuration
GAMMA = 0.9
DELTA_EPSILON = 0.00001
BATCH_SIZE = 32
LOOK_BACK = 4
OBSERVATION_THRESHOLD = 40
LISTEN_PORT=12345

# Constant
ACCELERATE = 0
BRAKE = 1
TURN_LEFT = 2
TURN_RIGHT = 3
NB_ACTION = 4
ACTIONS = [
  'ACCELERATE', 'BRAKE     ', 'LEFT      ', 'RIGHT     '
]

# Misc functions
def isBlack(state):
  ret = True
  for e in np.nditer(state):
    if e > 10:
      ret = False
  return ret;

def toNPArray(buf, width, height):
  arr = np.empty([height, width, 1])
  for y in range(0, height):
    for x in range(0, width):
      green = buf[y * width + x]
      arr[y][x][0] = green
  return arr

def combine(frames, height, width):
  blank = np.zeros([height, width, 1])
  while len(frames) < LOOK_BACK:
    frames.insert(0, blank)
  return np.dstack(frames)

#####################################
#      Deep Network Definition      #
#####################################
def createNN(inputShape, weightFilepath = None):
  imageInput = Input(shape=inputShape, name='imageInput')
  auxiliaryInput = Input(shape=(1,), name='auxiliaryInput')
  conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(imageInput)
  conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
  conv3 = Conv2D(64, (3, 3), activation='relu')(conv2)
  flatten = Flatten()(conv3)
  merge = Concatenate()([flatten, auxiliaryInput])
  dense1 = Dense(512, activation='relu')(merge)
  dense2 = Dense(NB_ACTION)(dense1)

  model = Model(inputs=[imageInput, auxiliaryInput], outputs=[dense2])

  model.compile(loss="mean_squared_error", optimizer="rmsprop")
  if weightFilepath is not None:
      model.load_weights(weightFilepath)

  return model


#####################################
#      Reinforcement Learning       #
#####################################
class Player:
  def __init__(self, args):
    self.episode = 0
    self.historyStepCounter = 0
    self.cumulatedReward = 0
    self.memory = []
    self.frames = []
    self.height = -1
    self.width = -1
    self.nn = None
    self.episodeStepCounter = 0
    self.lastState = None
    self.lastAction = BRAKE
    self.lastVelocity = 0
    self.args = args
    print self.args
    self.epsilon = self.args['epsilon']

  def reset(self):
    self.episode += 1
    print 'Episode #' + str(self.episode) + ', Memory size = ' + str(len(self.memory))
    self.episodeStepCounter = 0
    self.cumulatedReward = 0
    self.frames = []
    blank = np.zeros([self.height, self.width, 1])
    for i in range(0, LOOK_BACK - 1):
      self.frames.append(blank)
    self.lastState = np.zeros([self.height, self.width, LOOK_BACK])
    self.lastAction = ACCELERATE
    self.lastVelocity = 0

  def learnQ(self, reward, velocity, newState, isEnd):
    self.cumulatedReward += reward
    self.episodeStepCounter += 1
    self.historyStepCounter += 1
  
    if not self.args['inference_only']:
      if not isBlack(lastState):
        self.memory.append((self.lastState, self.lastAction, reward, self.lastVelocity, newState, velocity, isEnd))
  
      if len(self.memory) > self.args['memory_size']:
        self.memory.pop(0)
      if len(self.memory) > OBSERVATION_THRESHOLD:
        miniBatch = random.sample(self.memory, BATCH_SIZE)
        xImage = []
        xAuxiliary = []
        y = [] 
        for t in miniBatch:
          sLast, aLast, r, vLast, s, v, isTerminal = t
          xImage.append(sLast)
          xAuxiliary.append([vLast])
          update = r
          if not isTerminal:
            q = self.nn.predict([np.array([s]), np.array([[v]])])
            a = np.argmax(q[0])
            update += GAMMA * q[0][a]
          qLast = self.nn.predict([np.array([sLast]), np.array([[vLast]])])
          qLast[0][aLast] = update
          y.append(qLast.reshape(NB_ACTION, ))
        XImage = np.array(xImage)
        XAuxiliary = np.array(xAuxiliary)
        Y = np.array(y)
        self.nn.fit([XImage, XAuxiliary], Y, epochs=1, verbose=0)
  
    if isEnd:
      print 'Score = ' + str(self.cumulatedReward / float(self.episodeStepCounter))
      self.reset()
      return BRAKE
  
    if self.epsilon > self.args['min_epsilon']:
      self.epsilon -= DELTA_EPSILON
    
    # epsilon-greedy
    logQ = ''
    if random.random() < self.epsilon:
      action = np.random.randint(0, NB_ACTION) # Random exploration
    else:
      q = self.nn.predict([np.array([newState]), np.array([[velocity]])])
      action = np.argmax(q[0])
      logQ = ', action = ' + ACTIONS[action] + ', Q = ' + str(q[0][action])
    log = 'lastAction = ' + ACTIONS[self.lastAction]
    log += ', reward = ' + '%7.2f' % (reward)
    log += ', V = ' + '%10.5f' % (velocity)
    log += logQ
    print log
    self.lastAction = action
    self.lastState = newState
    self.lastVelocity = velocity
  
    if (self.historyStepCounter % 1000 == 0):
      print 'epsilon = ' + str(self.epsilon)
    if (not self.args['inference_only']) \
       and ('model_file' in self.args) \
       and self.historyStepCounter % 25000 == 0:
      self.nn.save_weights(self.args['model_file'], overwrite=True)
      print 'model saved'
  
    return action

  def readFrame(self, sock):
    header = sock.recv(struct.calcsize('IIffIII'))
    magic, frameCounter, reward, velocity, restart, width, height = struct.unpack('IIffIII', header)
    if magic != 0xdeadbeef:
      print 'Communication error'
      sys.exit(1)
    imageLen = struct.calcsize('%sB' % (width * height))
    image = bytearray()
    while imageLen > 0:
      buf = sock.recv(imageLen)
      imageLen -= len(buf)
      image += buf
    return (reward, velocity, restart, width, height, image)

  def play(self):
    try:
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error, msg:
      sys.stderr.write("[ERROR] %s\n" % msg[1])
      sys.exit(1)
    
    try:
      sock.connect(('localhost', LISTEN_PORT))
    except socket.error, msg:
      sys.stderr.write("[ERROR] %s\n" % msg[1])
      exit(1)
    
    print 'Simulator connected'
    
    combinedReward = 0
    while True:
      reward, velocity, ending, width, height, image = self.readFrame(sock)
      if self.nn is None:
        if 'model_file' in self.args:
          self.nn = createNN((height, width, LOOK_BACK), self.args['model_file'])
        else:
          self.nn = createNN((height, width, LOOK_BACK))
        self.height = height
        self.width = width
        self.reset()
    
      self.frames.append(toNPArray(image, width, height))
      combinedReward += reward
      if len(self.frames) >= LOOK_BACK or ending > 0:
        state = combine(self.frames, height, width)
        action = self.learnQ(combinedReward,
                             velocity * 1000, # *1000 for prettiness
                             state,
                             ending > 0)
        self.frames = []
        combinedReward = 0
      else:
        action = self.lastAction
      sock.send(struct.pack('I', action))
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Argument parser')
  parser.add_argument('-i', '--inference-only', action='store_true')
  parser.add_argument('-e', '--epsilon', type=float, default=1)
  parser.add_argument('--min-epsilon', type=float, default=0.1)
  parser.add_argument('--memory-size', type=int, default=50000)
  parser.add_argument('-m', '--model-file')
  args = parser.parse_args()

  player = Player(vars(args))
  player.play()

