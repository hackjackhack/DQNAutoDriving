#!/usr/bin/python

import direct.directbase.DirectStart
import signal
import socket
import struct
import sys

from math import pi
from math import cos
from math import sin
from threading import Thread
from threading import Event
from threading import Lock
from time import sleep
from panda3d.core import ClockObject
from panda3d.core import PNMImage
from panda3d.core import Vec2
from panda3d.core import Vec3
from panda3d.core import Texture
from panda3d.core import TextureStage
from panda3d.core import CardMaker
from panda3d.core import Camera
from panda3d.core import WindowProperties
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape

# Configuration
EPISODE_FRAME_LIMIT = 100000
DEVIATION_LIMIT = 1000
FRICTION = 0.0001
ACCELERATION_COEFF = 0.001
MAX_VELOCITY = 0.1
MIN_VELOCITY = -0.1
LISTEN_PORT = 12345
FRAME_RATE = 20

# CONSTANT
ACCELERATE = 0
BRAKE = 1
TURN_LEFT = 2
TURN_RIGHT = 3
PITCH = -45

# Misc functions
def clamp(val, floor, ceil):
  return max(min(val, ceil), floor)

class Simulator:
  # Class variables
  def __init__(self, mapFileName):
    # Instance variables
    self.camPos = Vec3(0, 0, 0)
    self.faceVec = Vec2(0, 0)
    self.velocity = 0
    self.episodeFrameCounter = 0
    self.historyFrameCounter = 0
    self.deviationCounter = 0
    self.accSignal = 0
    self.turnSignal = 0
    self.signalGameFeedback = Event()
    self.lock = Lock()
    self.bufferFeedback = bytearray()
    self.signalShutdown = Event()

    # Panda3D-related initialization
    # Car-view Camera
    base.camLens.setNear(0.05)
    base.camLens.setFov(70)

    # A fixed, top-view Camera
    _ = Camera("top_cam")
    topCam = render.attachNewNode(_)
    topCam.setName("top_cam")
    topCam.setPos(0, 0, 40)
    topCam.setHpr(0, -90, 0)
    
    dr = base.camNode.getDisplayRegion(0)
    dr.setActive(0)
    window = dr.getWindow()
    drTop = window.makeDisplayRegion(0.5, 1, 0, 0.5)
    drTop.setSort(dr.getSort())
    drTop.setCamera(topCam)
    self.drCar = window.makeDisplayRegion(0, 0.5, 0.5, 1)
    self.drCar.setSort(dr.getSort())
    self.drCar.setCamera(base.cam)

    # BulletWorld
    world = BulletWorld()
    world.setGravity(Vec3(0, 0, 0))
    props = WindowProperties() 
    props.setSize(1024, 768) 
    base.win.requestProperties(props)

    # Floor
    shapeGround = BulletPlaneShape(Vec3(0, 0, 1), 0.1)
    nodeGround = BulletRigidBodyNode('Ground')
    nodeGround.addShape(shapeGround)
    npGround = render.attachNewNode(nodeGround)
    npGround.setPos(0, 0, -2)
    world.attachRigidBody(nodeGround)
    
    cmGround = CardMaker("Ground")
    cmGround.setFrame(-10, 10, -10, 10);
    npGroundTex = render.attachNewNode(cmGround.generate())
    npGroundTex.reparentTo(npGround)
    ts = TextureStage("ts")
    ts.setMode(TextureStage.M_modulate)
    groundTexture = loader.loadTexture(mapFileName)
    npGroundTex.setP(270)
    npGroundTex.setTexScale(ts, 1, 1)
    npGround.setTexture(ts, groundTexture)

    # Car
    shapeBox = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
    nodeBox = BulletRigidBodyNode('Box')
    nodeBox.setMass(1.0)
    nodeBox.addShape(shapeBox)
    self.npBox = render.attachNewNode(nodeBox)
    self.npBox.setPos(self.camPos)
    world.attachRigidBody(nodeBox)
    modelBox = loader.loadModel('models/box.egg')
    modelBox.flattenLight()
    modelBox.setScale(0.5)
    modelBox.reparentTo(self.npBox)

  def reset(self):
    self.camPos = Vec3(-0.4, -9.2, -1.8)
    self.faceVec = Vec2(0, 1)
    self.velocity = 0
    base.cam.setPos(self.camPos)
    base.cam.setHpr(0, PITCH, 0)
    self.episodeFrameCounter = 0
    self.deviationCounter = 0
    self.accSignal = 0
    self.turnSignal = 0

  def calculateReward(self, screenshot):
    reward = 0
  
    # Penalty if wheels touch the lane
    width = screenshot.getXSize()
    height = screenshot.getYSize()
  
    leftBound = width * 3 / 7
    rightBound = width * 4 / 7
    upperBound = height * 6 / 7
    brightSum = 0
    for y in range(upperBound, height):
      for x in range(leftBound, rightBound):
        brightSum += screenshot.getBright(x, y)
    brightSum /= (rightBound - leftBound) * (height - upperBound)
  
    if brightSum > 0.1:
      if self.velocity <= 0:
        reward = brightSum * 100
      else:
        reward = brightSum * (1 + self.velocity) * 500
    else:
      reward = (abs(self.velocity) + 0.1)  * -1000
  
    if brightSum < 0.1:
      self.deviationCounter += 1
    return reward

  def update(self):
    self.historyFrameCounter += 1
    self.episodeFrameCounter += 1
    if self.episodeFrameCounter % 1000 == 0:
      print 'frame #' + str(self.episodeFrameCounter)

    if self.velocity > 0:
      self.velocity = max(0, self.velocity - FRICTION)
    elif self.velocity < 0:
      self.velocity = min(0, self.velocity + FRICTION)
    self.velocity = self.velocity + self.accSignal * ACCELERATION_COEFF
    self.velocity = clamp(self.velocity, MIN_VELOCITY, MAX_VELOCITY)
    self.turnSignal = clamp(self.turnSignal, -1, 1)
    theta = -1.0 * self.turnSignal / 360 * 2 * pi

    # Clear signal
    self.accSignal = 0
    self.turnSignal = 0

    fx = self.faceVec.getX()
    fy = self.faceVec.getY()
    self.faceVec = Vec2(fx * cos(theta) - fy * sin(theta), fx * sin(theta) + fy * cos(theta))
    self.faceVec.normalize()

    speedVec = self.faceVec * self.velocity
    self.camPos += Vec3(speedVec.getX(), speedVec.getY(), 0)

    deg = Vec2(0, 1).signedAngleDeg(self.faceVec)
    base.cam.setHpr(deg, PITCH, 0)
    base.cam.setPos(self.camPos)
    self.npBox.setPos(self.camPos)
    self.npBox.setHpr(deg, 0, 0)
    
    screenshot = PNMImage()
    if not self.drCar.getScreenshot(screenshot):
      return

    screenshot.removeAlpha()
    reward = self.calculateReward(screenshot)
    if self.camPos.getY() < -9.5 or self.camPos.getY() > 9.5 \
       or self.camPos.getX() < -7 or self.camPos.getX() > 7 \
       or self.episodeFrameCounter > EPISODE_FRAME_LIMIT \
       or self.deviationCounter > DEVIATION_LIMIT:
      self.reset()
    self.feedback(screenshot, reward, 1 if self.episodeFrameCounter == 0 else 0)
    return

  def feedback(self, screenshot, reward, isEnd):
    resized = PNMImage(80, 60)
    resized.unfilteredStretchFrom(screenshot)
    width = resized.getXSize()
    height = resized.getYSize()
    green = \
      map(lambda i: int(resized.getGreen(i % width, i / width) * 256) % 256, \
          range(0, width * height))
  
    data = struct.pack('IIffIII%sB' % len(green),
                       0xdeadbeef, self.historyFrameCounter,
                       reward, self.velocity,
                       isEnd, width, height, *green)
    
    with self.lock:
      self.bufferFeedback = data
    self.signalGameFeedback.set()
    # Make sure last frame of an episode is processed.
    if isEnd:
      sleep(1)

def communicationThread(simulator):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind(('', LISTEN_PORT))
  sock.listen(5)
  sock.settimeout(2)
  clientSock = None
  while not simulator.signalShutdown.is_set():
    try:
      (clientSock, address) = sock.accept()
      print 'Player connected'
      break
    except:
      pass

  while not simulator.signalShutdown.is_set():
    if not simulator.signalGameFeedback.wait(2):
      continue
    simulator.signalGameFeedback.clear()
    bufferLocal = bytearray()
    with simulator.lock:
      bufferLocal[:] = simulator.bufferFeedback
    clientSock.send(bufferLocal)
    action = struct.unpack('I', clientSock.recv(struct.calcsize('I')))[0]
    if action == ACCELERATE:
      simulator.accSignal = 1
      simulator.turnSignal = 0
    elif action == BRAKE:
      simulator.accSignal = -1
      simulator.turnSignal = 0
    elif action == TURN_LEFT:
      simulator.accSignal = 0
      simulator.turnSignal = -1
    elif action == TURN_RIGHT:
      simulator.accSignal = 0
      simulator.turnSignal = 1

simulator = None
def updateWrapper(task):
  global simulator
  simulator.update()
  return task.cont

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Usage: ./environment.py <map filename>'
    sys.exit(1)

  simulator = Simulator(sys.argv[1])
  simulator.reset()
  thread = Thread(target = communicationThread, args=[simulator])
  thread.start()

  globalClock.setMode(ClockObject.MLimited)
  globalClock.setFrameRate(FRAME_RATE)
  taskMgr.add(updateWrapper, 'update')

  # Start simulation.
  base.run()

  # Returning from thhe function above indicates that the simulation ends.
  # Terminate gracefully
  simulator.signalShutdown.set()
  thread.join()
