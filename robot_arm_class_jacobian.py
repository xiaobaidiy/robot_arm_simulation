import numpy as np
import math

class RobotArm3D:
    '''RobotArm3D([xRoot=0, yRoot=0, zRoot=0])

        INPUT ARGUMENTS:

        xRoot, yRoot, zRoot (optional): x y and z coordinates of the root joint.
            All default to 0 if not set.

        INSTANCE VARIABLES:

        thetas: 1D array of joint angles; contains N elements, one per joint.
        joints: 4 x N array of joint coordinates; each column is a vector
            (column 0 is the root joint and column N-1 is the end effector).
        a: list of arm link lengths, containing N elements, where
            lengths[0] is the first link and lengths[N-1] is the last link,
            terminating at the end effector.
        d: list of arm link lengths, containing N elements, where
            lengths[0] is the first link and lengths[N-1] is the last link,
            terminating at the end effector.
    '''
    def __init__(self, **kwargs):
        self.xRoot = kwargs.get('xRoot', 0)
        self.yRoot = kwargs.get('yRoot', 0)
        self.zRoot = kwargs.get('zRoot', 0)
        self.alphas = []
        self.thetas = np.array([[]], dtype=np.float)
        self.preThetas = np.array([[]], dtype=np.float)
        self.joints = np.array([[self.xRoot, self.yRoot, self.zRoot, 1]], dtype=np.float).T
        self.a = []
        self.d = []
        self.zUnitVec = np.array([[0,0,0]], dtype=np.float)
        self.xUnitVec = np.array([[0,0,0]], dtype=np.float)
        self.yUnitVec = np.array([[0,0,0]], dtype=np.float)

    def add_revolute_link(self, **kwargs):
        '''add_revolute_link(length[, thetaInit=0])
            Add a revolute joint to the arm with a link whose length is given
            by required argument "length". Optionally, the initial angle
            of the joint can be specified.
        '''
        self.joints = np.append(self.joints, np.array([[0,0,0,1]]).T, axis=1)
        self.a.append(kwargs['a'])
        self.d.append(kwargs['d'])
        self.alphas = np.append(self.alphas, kwargs.get('alphaInit', 0))
        self.thetas = np.append(self.thetas, kwargs.get('thetaInit', 0))
        self.preThetas = np.append(self.thetas, kwargs.get('thetaInit', 0))
        self.zUnitVec = np.append(self.zUnitVec, np.array([[0,0,0]]), axis=0)
        self.xUnitVec = np.append(self.xUnitVec, np.array([[0,0,0]]), axis=0)
        self.yUnitVec = np.append(self.yUnitVec, np.array([[0,0,0]]), axis=0)

    def get_transformation_matrix(self, alpha, theta, a, d):
        '''get_transformation_matrix(theta, x, y)
            Returns a 4x4 transformation matrix for a 2D rotation
            and translation. "theta" specifies the rotation. "x"
            and "y" specify the translational offset.
        '''
        transformationMatrix = np.array([
            [math.cos(theta), -math.sin(theta), 0, a],
            [math.sin(theta)*math.cos(alpha), math.cos(theta)*math.cos(alpha), -math.sin(alpha), -math.sin(alpha)*d],
            [math.sin(theta)*math.sin(alpha), math.cos(theta)*math.sin(alpha), math.cos(alpha), math.cos(alpha)*d],
            [0, 0, 0, 1]
            ])
        return transformationMatrix

    def update_joint_coords(self):
        '''update_joint_coords()
            Recompute x, y, z coordinates of each joint and end effector.
        '''
        
        # "T" is a cumulative transformation matrix that is the result of
        # the multiplication of all transformation matrices up to and including
        # the ith joint of the for loop.
        T = self.get_transformation_matrix(
            self.alphas[0], self.thetas[0].item(), self.a[0], self.d[0])
        self.zUnitVec[0] = T[:3, 2]
        self.xUnitVec[0] = T[:3, 0]
        self.yUnitVec[0] = T[:3, 1]
        for i in range(len(self.alphas) - 1):
            T_next = self.get_transformation_matrix(
                self.alphas[i+1], self.thetas[i+1].item(), self.a[i+1], self.d[i+1])
            T = T.dot(T_next)
            self.zUnitVec[i+1] = T[:3, 2]
            self.xUnitVec[i+1] = T[:3, 0]
            self.yUnitVec[i+1] = T[:3, 1]
            self.joints[:,[i+1]] = T.dot(np.array([[0,0,0,1]]).T)

        # Update end effector coordinates.
        endEffectorCoords = np.array([[1,0,0,5],
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])

        T = T.dot(endEffectorCoords)
        self.joints[:,[-1]] = T.dot(np.array([[0,0,0,1]]).T)

        # end effector "z-hat" is same with last joint
        self.zUnitVec[-1] = T[:3, 2]
        self.xUnitVec[-1] = T[:3, 0]
        self.yUnitVec[-1] = T[:3, 1]

    def get_jacobian(self):
        '''get_jacobian()
            Return the 6 x N Jacobian for the current set of joint angles.
        '''

        jacobian = np.zeros((6, len(self.joints[0,:]) - 1), dtype=np.float)
        endEffectorCoords = self.joints[:3,[-1]]

        # Utilize cross product to compute each row of the Jacobian matrix.
        for i in range(len(self.joints[0,:]) - 1):
            currentJointCoords = self.joints[:3,[i]]
            jacobian[:3,i] = np.cross(
                self.zUnitVec[i], (endEffectorCoords - currentJointCoords).reshape(3,))
            jacobian[3:,i] = self.zUnitVec[i].reshape(3,)
        return jacobian


    def update_theta(self, deltaTheta):
        self.thetas += deltaTheta.flatten()