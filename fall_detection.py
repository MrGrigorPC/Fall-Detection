import numpy as np

skeleton_1 = np.load('./data/skeleton_3.npy')


class FallDetecion:
    def __init__(self, fps=18):
        self.torso_up = np.array(
            [5, 6]
        )  # The slice used for generating the midpoint of the shoulders

        self.torso_down = np.array(
            [11, 12]
        )  # The slice used for generating the midpoint of the hips

        self.vector_indices = np.array(
            [
                [19, 17],
                [19, 18],
                [6, 12],
                [5, 11],
                [6, 8],
                [5, 7],
                [12, 14],
                [11, 13],
                [11, 12],
                [13, 15],
                [14, 16],
                [20, 21],
            ]
        )  # Vectors to be considered for calculating angles

        self.pair_indices = np.array(
            [[4, 2], [5, 3], [6, 10], [7, 9], [8, 6], [8, 7], [0, 11], [1, 11]]
        )  # The pairs of vectors for angle computation

        self.vertical_coordinates = np.array(
            [[1, 1], [1, 100]]
        )  # A vertical vector for comparing with other vectors

        self.fps = fps

    def angleCalculation(self, vectors):
        """
        Used to calculate the angles between given pairs of vectors
        Takes as input the list of vector pairs, which represent two vectors with two coordinates each
        Returns the list of angles between them
        """

        difference = np.subtract(
            vectors[:, :, :, 0], vectors[:, :, :, 1]
        )  # Subtracts the coordinates to obtain the vectors
        dot = (difference[:, :, 0, :] * difference[:, :, 1, :]).sum(
            axis=-1
        )  # Calculates the dot product between the pairs of vectors
        norm = np.prod(
            np.linalg.norm(difference[:, :, :, :], axis=3), axis=-1
        )  # Calculates the norm of the vectors and multiplies them, same as |a|*|b|

        cos_angle = np.divide(dot, norm)  # cos(x) = dot(a,b)/|a|*|b|

        angle = (
                np.arccos(cos_angle) * 180 / np.pi
        )  # Take arccos of the result to get the angle

        angle = angle.reshape(-1, 8, 1)  # Correct the shape of the output

        return angle

    def collectData(self, keypoints):
        """
        Used for handling negative predictions and adding extra points to the keypoints
        Takes as input the list of keypoints
        Returns the list of handled keypoints and added extra points
        """

        keypoints[keypoints < 0] = np.nan
        # Handle missing values
        torso_up = keypoints[:, self.torso_up].mean(axis=1)[:,
                   np.newaxis]  # Getting middle point of torso down and adding new axis for further concatenation
        torso_down = keypoints[:, self.torso_down].mean(axis=1)[:,
                     np.newaxis]  # Getting middle point of torso up and adding new axis for further concatenation
        head_coordinate = np.nanmean(keypoints[:, :5], axis=1)[:,
                          np.newaxis]  # Getting mean value of head coordinate and adding new axis for further concatenation
        vertical = np.tile(self.vertical_coordinates,
                           (keypoints.shape[0], 1, 1))  # Numpy broadcasting for further concatenation
        keypoints = np.concatenate((
            keypoints,
            torso_up,
            torso_down,
            head_coordinate,
            vertical
        ), axis=1)
        return keypoints

    def __call__(self, skeleton):
        '''
            This __call__ function takes a cache of skeletons as input, with a shape of (M x 17 x 2), where M represents the number of skeletons.
            The value of M is constant and represents time. For example, if you have a 7 fps stream and M is equal to 7 (M = 7), it means that the cache length is 1 second.
            The number 17 represents the count of points in each skeleton (as shown in skeleton.png), and 2 represents the (x, y) coordinates.

            This function uses the cache to detect falls.

            The function will return:
                - bool: isFall (True or False)
                - float: fallScore
        '''

        isFall = False
        keypoints = self.collectData(skeleton)
        vector_pairs = keypoints[:, self.vector_indices][:, self.pair_indices]
        angles = self.angleCalculation(vector_pairs)
        for i in range(len(angles) - 27):
            fallScore = np.nanmean(
                np.abs(np.diff(angles[i:i + 26], axis=0)) * self.fps)  # Making prediction using 27 frames(1.5 sec)
            if fallScore > 125:
                isFall = True
                return isFall, fallScore
        return isFall, fallScore


obj = FallDetecion()
print(obj(skeleton_1))
