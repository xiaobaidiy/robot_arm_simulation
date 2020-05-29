import numpy as np
import math
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from robot_arm_class_jacobian import *
import tkinter as tk 
import sys


def stopBtn():
    global stop
    stop = True

def startBtn():
    global stop, arrived, target, targetw
    arrived = False
    stop = False
    xInput = int(texts[0].get())
    yInput = int(texts[1].get())
    zInput = int(texts[2].get())
    wx = np.radians(int(texts[3].get()))
    wy = np.radians(int(texts[4].get()))
    wz = np.radians(int(texts[5].get()))

    targetPt.set_data_3d(xInput, yInput, zInput)
    target = np.array([[xInput, yInput, zInput, 1]]).T
    targetw = [wx, wy, wz]

    while arrived == False and stop == False:
        move_to_target()
        root.update()

def drawGUI():
    tk.Label(root, font = ('arial', 10, 'bold'), text = 'x:', fg = "green").grid(row = 0, column = 0)
    tk.Entry(root, width = 8, textvariable = texts[0]).grid(row = 0, column = 1)
    tk.Label(root, font = ('arial', 10, 'bold'), text = 'y:', fg = "green").grid(row = 0, column = 2)
    tk.Entry(root, width = 8, textvariable = texts[1]).grid(row = 0, column = 3)
    tk.Label(root, font = ('arial', 10, 'bold'), text = 'z:', fg = "green").grid(row = 0, column = 4)
    tk.Entry(root, width = 8, textvariable = texts[2]).grid(row = 0, column = 5)
    tk.Label(root, font = ('arial', 10, 'bold'), text = 'wx:', fg = "green").grid(row = 1, column = 0)
    tk.Entry(root, width = 8, textvariable = texts[3]).grid(row = 1, column = 1)
    tk.Label(root, font = ('arial', 10, 'bold'), text = 'wy:', fg = "green").grid(row = 1, column = 2)
    tk.Entry(root, width = 8, textvariable = texts[4]).grid(row = 1, column = 3)
    tk.Label(root, font = ('arial', 10, 'bold'), text = 'wz:', fg = "green").grid(row = 1, column = 4)
    tk.Entry(root, width = 8, textvariable = texts[5]).grid(row = 1, column = 5)
    button1 = tk.Button(master=root, text='Start', command=startBtn).grid(row=2, column=0)
    button2 = tk.Button(master=root, text='Stop', command=stopBtn).grid(row=2, column=2)
    button3 = tk.Button(master=root, text='Quit', command=sys.exit).grid(row=2, column=4)


def update_plot():
    '''Update arm and end effector line objects with current x and y
        coordinates from arm object.
    '''
    global arrow_x, arrow_y, arrow_z
    armLine.set_data_3d(Arm.joints[0,:-1], Arm.joints[1,:-1], Arm.joints[2,:-1])
    endEff.set_data_3d(Arm.joints[0,-2:], Arm.joints[1,-2:], Arm.joints[2,-2:])
    arrow_z.remove()
    arrow_x.remove()
    arrow_y.remove()
    arrow_z = ax.quiver(Arm.joints[0,-1], Arm.joints[1,-1], Arm.joints[2,-1], Arm.zUnitVec[-1,0], Arm.zUnitVec[-1,1], Arm.zUnitVec[-1,2], length=10, normalize=True)
    arrow_x = ax.quiver(Arm.joints[0,-1], Arm.joints[1,-1], Arm.joints[2,-1], Arm.xUnitVec[-1,0], Arm.xUnitVec[-1,1], Arm.xUnitVec[-1,2], length=10, normalize=True, color='r')
    arrow_y = ax.quiver(Arm.joints[0,-1], Arm.joints[1,-1], Arm.joints[2,-1], Arm.yUnitVec[-1,0], Arm.yUnitVec[-1,1], Arm.yUnitVec[-1,2], length=10, normalize=True, color='k')

def getEndAngle():
    R = np.array([Arm.xUnitVec[-1], Arm.yUnitVec[-1], Arm.zUnitVec[-1]])
    if R[2,0] > 0.98:
        beta = -0.5*np.pi                   #Y
        alpha = 0                           #Z
        gamma = -np.arctan2(R[0,1], R[1,1]) #X
    elif R[2,0] < -0.98:
        beta = 0.5*np.pi
        alpha = 0
        gamma = np.arctan2(R[0,1], R[1,1])
    else:
        beta = np.arctan2(-R[2,0],np.sqrt(R[0,0]**2+R[1,0]**2))
        alpha = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        gamma = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))

    return gamma, beta, alpha #X, Y, Z


def move_to_target():
    '''Run Jacobian inverse routine to move end effector toward target.'''

    # Set distance to move end effector toward target per algorithm iteration.
    global Arm, target, reach, arrived, targetw

    distPerUpdate = 0.02 * reach
    angPerUpdate = 0.08
    condition_pos = np.linalg.norm(target - Arm.joints[:,[-1]])
    gamma, beta, alpha = getEndAngle()
    condition_ang = np.linalg.norm(targetw - np.array([gamma, beta, alpha]))

    if  condition_pos < distPerUpdate and condition_ang < angPerUpdate :
        arrived = True
    else:
        targetVector = (target - Arm.joints[:,[-1]])[:3]
        targetUnitVector = targetVector / np.linalg.norm(targetVector)
        deltaR = distPerUpdate * targetUnitVector
        
        delta = np.array([deltaR[0][0], deltaR[1][0], deltaR[2][0], -angPerUpdate*(targetw[0]-gamma), -angPerUpdate*(targetw[1]-beta), -angPerUpdate*(targetw[2]-alpha)]).T
        
        J = Arm.get_jacobian()
        
        JInv = np.linalg.pinv(J)
        
        deltaTheta = JInv.dot(delta)

        Arm.update_theta(deltaTheta)

        Arm.update_joint_coords()
        update_plot()
        time.sleep(0.1)

arrived = False
stop = False
# Instantiate robot arm class.
Arm = RobotArm3D()

# #D-H table
# #1   90     0       0     theta1 -90 ----  90
# #2   -90    0       0     theta2 -90 ---- -270
# #3   -90    0       d3    theta3 -180 ---  0
# #4   -90    0       0     theta4  0  ---- 180
# #5   90     0       d5    theta5 -90 ---- 90
# #6   -90    0       0     theta6  0  ---- -180
# #7   -90    a6       0    theta7 -90 ---- 90
# # Add desired number of joints/links to robot arm object.
Arm.add_revolute_link(a=0, d=0, alphaInit=math.radians(90) ,thetaInit=math.radians(-90))
Arm.add_revolute_link(a=0, d=0, alphaInit=math.radians(-90) ,thetaInit=math.radians(-90))
Arm.add_revolute_link(a=0, d=20, alphaInit=math.radians(-90) ,thetaInit=math.radians(-90))
Arm.add_revolute_link(a=0, d=0, alphaInit=math.radians(-90) ,thetaInit=math.radians(0))
Arm.add_revolute_link(a=0, d=20, alphaInit=math.radians(90) ,thetaInit=math.radians(0))
Arm.add_revolute_link(a=0, d=0, alphaInit=math.radians(-90) ,thetaInit=math.radians(-90))
Arm.add_revolute_link(a=5, d=0, alphaInit=math.radians(-90) ,thetaInit=math.radians(0))
Arm.update_joint_coords()

# Initialize target coordinates to current end effector position.
target = Arm.joints[:,[-1]]
targetw = np.array([0, 0, 0])

root = tk.Tk()
root.wm_title("Robot arm control panel")

texts = []
for i in range(3):
    texts.append(tk.StringVar())
    texts[i].set(int(target[i][0]))
initwx, initwy, initwz = getEndAngle()
for i in range(3):
    texts.append(tk.StringVar())

texts[3].set(int(np.degrees(initwx)))
texts[4].set(int(np.degrees(initwy)))
texts[5].set(int(np.degrees(initwz)))

drawGUI()

# Initialize plot and line objects for target, end effector, and arm.
fig = plt.figure()
ax = plt.axes(projection='3d')
targetPt, = ax.plot3D([], [], [], marker='o', c='r')
endEff, = ax.plot3D([], [], [], marker='o', markerfacecolor='w', c='g', lw=1)
armLine, = ax.plot3D([], [], [], marker='o', c='g', lw=1)
base_arrow_z = ax.quiver([0], [0], [0], [0], [0], [1], length=10, normalize=True)
base_arrow_x = ax.quiver([0], [0], [0], [1], [0], [0], length=10, normalize=True, color='r')
base_arrow_y = ax.quiver([0], [0], [0], [0], [1], [0], length=10, normalize=True, color='k')
arrow_z = ax.quiver([], [], [], [], [], [], length=10, normalize=True)
arrow_x = ax.quiver([], [], [], [], [], [], length=10, normalize=True, color='r')
arrow_y = ax.quiver([], [], [], [], [], [], length=10, normalize=True, color='k')
# circle, = ax.plot3D([], [], [], color = 'k', ls='--', lw=1)
reach = 50

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Set axis limits based on reach from root joint.
ax.set_xlim(Arm.xRoot - 1.2 * reach, Arm.xRoot + 1.2 * reach)
ax.set_ylim(Arm.yRoot - 1.2 * reach, Arm.yRoot + 1.2 * reach)
ax.set_zlim(Arm.zRoot - 1.2 * reach, Arm.zRoot + 1.2 * reach)

# t = np.linspace(0, 2*np.pi, 100)
# circle.set_data_3d(30, 10 * np.sin(t) - 20, 10 * np.cos(t) - 20 );

update_plot()

plt.ion()
plt.show()

root.mainloop()