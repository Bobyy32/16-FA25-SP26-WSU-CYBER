def __eq__(self, other):
        """Check if two KeypointsOnImage instances are equal.

        Added in 0.4.0.

        Parameters
        ----------
        other : KeypointsOnImage or Keypoint, optional
            The object to compare to. If this is a ``Keypoint``, it will
            be compared to each keypoint in this ``KeypointsOnImage``.

        Returns
        -------
        bool
            ``True`` if all keypoints in both instances are equal.

        """
        if isinstance(other, KeypointsOnImage):
            return (len(self.keypoints) == len(other.keypoints) and
                    self._compare_keypoints(other))
        return False

    def __ne__(self, other):
        """Check if two KeypointsOnImage instances are not equal.

        Added in 0.4.0.

        Parameters
        ----------
        other : KeypointsOnImage or Keypoint, optional
            The object to compare to. If this is a ``Keypoint``, it will
            be compared to each keypoint in this ``KeypointsOnImage``.

        Returns
        -------
        bool
            ``True`` if any keypoint in this instance is different from
            the other.

        """
        return not self.__eq__(other)

    def _compare_keypoints(self, other):
        """Compare keypoints from two instances.

        Added in 0.4.0.

        Parameters
        ----------
        other : KeypointsOnImage
            The other instance to compare to.

        Returns
        -------
        bool
            ``True`` if all keypoints in this instance are equal to the
            corresponding keypoints in the other instance.

        """
        if len(self.keypoints) != len(other.keypoints):
            return False
        for kp_self, kp_other in zip(self.keypoints, other.keypoints):
            if (kp_self.x != kp_other.x or
                kp_self.y != kp_other.y):
                return False
        return True

    def __add__(self, other):
        """Add another ``KeypointsOnImage`` to this one.

        This method performs a shallow copy of the keypoints from the other
        instance and adds them to this one.

        Added in 0.4.0.

        Parameters
        ----------
        other : KeypointsOnImage
            The keypoints to add.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            A new ``KeypointsOnImage`` object with combined keypoints.

        """
        return self._combine_with(other)

    def __sub__(self, other):
        """Subtract another ``KeypointsOnImage`` from this one.

        This method performs a deep copy of the keypoints from the other
        instance and creates a new instance with the difference.

        Added in 0.4.0.

        Parameters
        ----------
        other : KeypointsOnImage
            The keypoints to subtract.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            A new ``KeypointsOnImage`` object with combined keypoints.

        """
        return self._combine_with(other)

    def __contains__(self, keypoint):
        """Check if a keypoint is in this container.

        Parameters
        ----------
        keypoint : Keypoint
            The keypoint to check.

        Returns
        -------
        bool
            ``True`` if the keypoint is present.

        """
        for kp in self.keypoints:
            if (kp.x == keypoint.x and kp.y == keypoint.y):
                return True
        return False

    def __getitem__(self, idx):
        """Get the keypoint at the given index.

        Parameters
        ----------
        idx : int
            The index of the keypoint to retrieve.

        Returns
        -------
        Keypoint
            The keypoint at the given index.

        """
        return self.keypoints[idx]

    def set_item(self, idx, kp):
        """Set a keypoint at the given index.

        Added in 0.4.0.

        Parameters
        ----------
        idx : int
            The index of the keypoint to set.
        kp : Keypoint
            The keypoint to set.

        """
        self.keypoints[idx] = kp

    def delete_item(self, idx):
        """Delete a keypoint at the given index.

        Added in 0.4.0.

        Parameters
        ----------
        idx : int
            The index of the keypoint to delete.

        Returns
        -------
        None

        """
        del self.keypoints[idx]

    def set_shape(self, shape):
        """Set the shape of the image on which keypoints are placed.

        Added in 0.4.0.

        Parameters
        ----------
        shape : tuple of int
            The shape of the image, ``(height, width)``.

        Returns
        -------
        KeypointsOnImage
            The ``KeypointsOnImage`` object with updated shape.

        """
        self.shape = shape
        return self

    def to_numpy(self):
        """Convert the keypoints to a NumPy array.

        Added in 0.4.0.

        Returns
        -------
        numpy.ndarray
            An array of shape ``(n_points, 2)`` or ``(n_points, nb_channels, 2)``,
            where each row is a keypoint's coordinates.

        """
        arr = np.zeros((len(self.keypoints), 2))
        for idx, kp in enumerate(self.keypoints):
            arr[idx, 0] = kp.x
            arr[idx, 1] = kp.y
        return arr

    def get_coordinates(self):
        """Get coordinates of all keypoints.

        Added in 0.4.0.

        Returns
        -------
        list of tuple
            A list of ``(x, y)`` tuples for each keypoint.

        """
        return [(kp.x, kp.y) for kp in self.keypoints]

    def set_coordinates(self, coords):
        """Set coordinates of all keypoints.

        Parameters
        ----------
        coords : list of tuple
            A list of ``(x, y)`` tuples for each keypoint.

        Returns
        -------
        KeypointsOnImage
            The ``KeypointsOnImage`` object with updated coordinates.

        """
        new_keypoints = [Keypoint(x=coord[0], y=coord[1]) for coord in coords]
        self.keypoints = new_keypoints
        return self

    def __delitem__(self, idx):
        """Delete the keypoint at the given index.

        Parameters
        ----------
        idx : int
            The index of the keypoint to delete.

        Returns
        -------
        None

        """
        del self.keypoints[idx]

    def __iter__(self):
        """Iterate over the keypoints.

        Yields
        -------
        Keypoint
            Each keypoint in the list.

        """
        for kp in self.keypoints:
            yield kp

    def __len__(self):
        """Return the number of keypoints.

        Returns
        -------
        int
            The number of keypoints.

        """
        return len(self.keypoints)