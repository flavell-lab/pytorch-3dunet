import importlib

import math
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, affine_transform
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from torchvision.transforms import Compose

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
#GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, axes=[(1,2)], force_180=False, **kwargs):
        self.random_state = random_state
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.axes = axes
        self.force_180 = force_180

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        if self.force_180:
            k = 2 * self.random_state.randint(0, 2)
        else:
            k = self.random_state.randint(0, 4)

        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, axis)
        else:
            channels = [np.rot90(m[c], k, axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m
        


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, cval=None, execution_probability=0.1, **kwargs):
        if axes is None:
            axes = [(0,1), (1,2), (0,2)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.execution_probability = execution_probability
        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order
        self.cval = cval

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            if self.cval is None:
                cval = np.median(m)
            else:
                cval = self.cval
            axis = self.axes[self.random_state.randint(len(self.axes))]
            angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)
            if m.ndim == 3:
                m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=cval)
            else:
                channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=cval) for c
                            in range(m.shape[0])]
                m = np.stack(channels, axis=0)

        return m


class RandomAffineTransform:
    """ Implements a random affine transform. Includes shear, scaling, rotation, and (TODO) translation.
        Order is shear, then scale, then rotation. Only rotates in xy plane."""
    def __init__(self, random_state, sigma_xy_shear=0.1, sigma_zstack_shear=0.1, sigma_zwarp_shear=0.1, sigma_scale_xy=0.1, sigma_scale_z=0.1, sigma_rotate=15, axes=None,
            shear_exec_prob=0.2, rotate_exec_prob=0.2, scale_exec_prob=0.2, translate_exec_prob=0.4, translate_x=50, translate_y=20, translate_z=10, mode='constant', order=1, cval=None, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.random_state = random_state
        self.sigma_xy_shear = sigma_xy_shear
        self.sigma_zstack_shear = sigma_zstack_shear
        self.sigma_zwarp_shear = sigma_zwarp_shear
        self.sigma_scale_xy = sigma_scale_xy
        self.sigma_scale_z = sigma_scale_z
        self.sigma_rotate = sigma_rotate
        self.shear_exec_prob = shear_exec_prob
        self.rotate_exec_prob = rotate_exec_prob
        self.scale_exec_prob = scale_exec_prob
        self.order = order
        self.mode = mode
        self.cval = cval
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.translate_z = translate_z
        self.translate_exec_prob = translate_exec_prob
        self.axes = axes

    def __call__(self, m):
        assert m.ndim == 3
        mat_shear = np.identity(3)
        if self.random_state.uniform() < self.shear_exec_prob:
            if self.sigma_xy_shear > 0:
                mat_shear[2,1] = self.random_state.normal(0,self.sigma_xy_shear)
                mat_shear[1,2] = self.random_state.normal(0,self.sigma_xy_shear)
            if self.sigma_zstack_shear > 0:
                mat_shear[2,0] = self.random_state.normal(0,self.sigma_zstack_shear)
                mat_shear[1,0] = self.random_state.normal(0,self.sigma_zstack_shear)
            if self.sigma_zwarp_shear > 0:
                mat_shear[0,1] = self.random_state.normal(0,self.sigma_zwarp_shear)
                mat_shear[0,2] = self.random_state.normal(0,self.sigma_zwarp_shear)

        mat_scale = np.identity(3)

        if self.random_state.uniform() < self.scale_exec_prob:
            if self.sigma_scale_xy > 0:
                mat_scale[1,1] = self.random_state.normal(1,self.sigma_scale_xy)
                mat_scale[2,2] = self.random_state.normal(1,self.sigma_scale_xy)
            if self.sigma_scale_z > 0:
                mat_scale[0,0] = self.random_state.normal(1,self.sigma_scale_z)

        mat_rotate = np.identity(3)
        if self.random_state.uniform() < self.rotate_exec_prob:
            theta = self.random_state.normal(0, self.sigma_rotate)
            axis = self.axes[self.random_state.randint(len(self.axes))]
            mat_rotate[axis[1],axis[1]] = math.cos(math.radians(theta))
            mat_rotate[axis[1],axis[0]] = -math.sin(math.radians(theta))
            mat_rotate[axis[0],axis[1]] = math.sin(math.radians(theta))
            mat_rotate[axis[0],axis[0]] = math.cos(math.radians(theta))

        mat = np.dot(np.dot(mat_rotate, mat_scale), mat_shear)
        if self.cval is None:
            cval = np.median(m)
        else:
            cval = self.cval

        offset = [0,0,0]
        if self.random_state.uniform() < self.translate_exec_prob:
            offset[0] = self.random_state.randint(-self.translate_z, self.translate_z)
            offset[1] = self.random_state.randint(-self.translate_y, self.translate_y)
            offset[2] = self.random_state.randint(-self.translate_x, self.translate_x)

        m = affine_transform(m, mat, offset=offset, mode=self.mode, order=self.order, cval=cval)

        return m

class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    For 4D input (CxDxHxW), it applies the adjustment independently for each channel.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability
        self.use_data_mean = kwargs.get('use_data_mean', False)

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            if m.ndim == 3:  # For 3D data
                return self.adjust_contrast(m)
            elif m.ndim == 4:  # For 4D data
                return np.array([self.adjust_contrast(channel) for channel in m])
        return m

    def adjust_contrast(self, data):
        if self.use_data_mean:
            mu = np.mean(data)
        else:
            mu = self.mean
        alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
        result = mu + alpha * (data - mu)
        return np.clip(result, -1, 1)


class RandomITKDeformation:
    """ Implements a random ITK transform. Includes shear, scaling, rotation, translation, and B-Spline.
        Order is B-Spline, then shear, then scale, then rotation."""
    def __init__(self, random_state, sigma_xy_shear=0.1, sigma_zstack_shear=0.1, sigma_zwarp_shear=0.1, sigma_scale_xy=0.1, sigma_scale_z=0.1, sigma_xy_rotate=15, sigma_xz_rotate=1, sigma_yz_rotate=5,
            shear_exec_prob=0.2, xy_rotate_exec_prob=0.2, xz_rotate_exec_prob=0.2, yz_rotate_exec_prob=0.2, scale_exec_prob=0.2, translate_exec_prob=0.4, translate_x=50, translate_y=20, translate_z=10, mode='constant', order=1, cval=None,
            bspline_exec_prob=0.2, interpolator='linear', bspline_order=2, bspline_ctrl_points_x=3, bspline_bend_lim=0.35, bspline_sigma=5, **kwargs):
        self.random_state = random_state
        self.sigma_xy_shear = sigma_xy_shear
        self.sigma_zstack_shear = sigma_zstack_shear
        self.sigma_zwarp_shear = sigma_zwarp_shear
        self.sigma_scale_xy = sigma_scale_xy
        self.sigma_scale_z = sigma_scale_z
        self.sigma_xy_rotate = sigma_xy_rotate
        self.sigma_xz_rotate = sigma_xz_rotate
        self.sigma_yz_rotate = sigma_yz_rotate
        self.shear_exec_prob = shear_exec_prob
        self.xy_rotate_exec_prob = xy_rotate_exec_prob
        self.xz_rotate_exec_prob = xz_rotate_exec_prob
        self.yz_rotate_exec_prob = yz_rotate_exec_prob
        self.scale_exec_prob = scale_exec_prob
        self.cval = cval
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.translate_z = translate_z
        self.translate_exec_prob = translate_exec_prob

        self.bspline_exec_prob = bspline_exec_prob
        self.bspline_order = bspline_order
        self.bspline_ctrl_points_x = bspline_ctrl_points_x
        self.bspline_bend_lim = bspline_bend_lim
        self.bspline_sigma = bspline_sigma

        assert self.bspline_order == 2, "ERROR: only B-Spline order 2 is supported."
        if interpolator == 'bspline':
            self.interpolator = sitk.sitkBSpline
        elif interpolator == 'linear':
            self.interpolator = sitk.sitkLinear
        elif interpolator == 'nn':
            self.interpolator = sitk.sitkNearestNeighbor
        else:
            raise NotImplementedError("ERROR: interpolator " + str(interpolator) + " not implemented.")

    def _apply_transformation(self, m, restore_rng_state=False):
        # If restore_rng_state is True, save the current state of the random number generator
        if restore_rng_state:
            rng_state = self.random_state.get_state()

        mat_shear = np.identity(3)
        # SimpleITK pararmeters are in (x,y,z) order instead of (z,y,x) order
        if self.random_state.uniform() < self.shear_exec_prob:
            if self.sigma_xy_shear > 0:
                mat_shear[0,1] = self.random_state.normal(0,self.sigma_xy_shear)
                mat_shear[1,0] = self.random_state.normal(0,self.sigma_xy_shear)
            if self.sigma_zstack_shear > 0:
                mat_shear[0,2] = self.random_state.normal(0,self.sigma_zstack_shear)
                mat_shear[1,2] = self.random_state.normal(0,self.sigma_zstack_shear)
            if self.sigma_zwarp_shear > 0:
                mat_shear[2,1] = self.random_state.normal(0,self.sigma_zwarp_shear)
                mat_shear[2,0] = self.random_state.normal(0,self.sigma_zwarp_shear)

        mat_scale = np.identity(3)

        if self.random_state.uniform() < self.scale_exec_prob:
            if self.sigma_scale_xy > 0:
                mat_scale[1,1] = self.random_state.normal(1,self.sigma_scale_xy)
                mat_scale[0,0] = self.random_state.normal(1,self.sigma_scale_xy)
            if self.sigma_scale_z > 0:
                mat_scale[2,2] = self.random_state.normal(1,self.sigma_scale_z)

        offset = [0,0,0]

        rotate_probs = [self.xy_rotate_exec_prob, self.xz_rotate_exec_prob, self.yz_rotate_exec_prob]
        rotate_axes = [(0,1), (0,2), (1,2)]
        rotate_sigmas = [self.sigma_xy_rotate, self.sigma_xz_rotate, self.sigma_yz_rotate]
        centroid = np.array([m.shape[2] / 2, m.shape[1] / 2, m.shape[0] / 2])

        mat_rotate = np.identity(4)  # Make it 4x4 for homogeneous coordinates


        for dim in range(3):
            if self.random_state.uniform() < rotate_probs[dim]:
                # Step 1: Translate to move centroid to origin
                translate_to_origin = np.identity(4)
                translate_to_origin[:3, 3] = -centroid

                # Your existing rotation code here
                axis = rotate_axes[dim]

                theta = self.random_state.normal(0, rotate_sigmas[dim])

                mat_rotate_axis = np.identity(4)  # Make it 4x4 for homogeneous coordinates

                mat_rotate_axis[axis[1],axis[1]] = math.cos(math.radians(theta))
                mat_rotate_axis[axis[1],axis[0]] = -math.sin(math.radians(theta))
                mat_rotate_axis[axis[0],axis[1]] = math.sin(math.radians(theta))
                mat_rotate_axis[axis[0],axis[0]] = math.cos(math.radians(theta))

                translate_back = np.identity(4)
                translate_back[:3, 3] = centroid

                combined_transform = np.dot(np.dot(translate_back, mat_rotate_axis), translate_to_origin)

                mat_rotate = np.dot(combined_transform, mat_rotate)

        offset += mat_rotate[:3, 3]
        mat_rotate = mat_rotate[:3, :3]

        mat = np.dot(np.dot(mat_rotate, mat_scale), mat_shear)
        if self.cval is None:
            cval = np.median(m)
        else:
            cval = self.cval

        if self.random_state.uniform() < self.translate_exec_prob:
            offset[2] += self.random_state.randint(-self.translate_z, self.translate_z)
            offset[1] += self.random_state.randint(-self.translate_y, self.translate_y)
            offset[0] += self.random_state.randint(-self.translate_x, self.translate_x)

        aff_tfm = sitk.AffineTransform(3)
        aff_tfm.SetMatrix(mat.flatten())
        aff_tfm.SetTranslation(offset)
        raw_img = sitk.GetImageFromArray(m)
        t = aff_tfm
        if self.random_state.uniform() < self.bspline_exec_prob:
            n_ctrl_points_x = self.bspline_ctrl_points_x
            bspline_order = self.bspline_order
            sigma = self.bspline_sigma
            bsp_grid_size = [n_ctrl_points_x, 1, 1]
            bspline_tfm = sitk.BSplineTransformInitializer(raw_img, bsp_grid_size, order=bspline_order)
            params = np.asarray(bspline_tfm.GetParameters(), dtype=np.float64)

            x_params = params[0:len(params)//3]  # Assuming the parameters are ordered zyx
            y_params = params[len(params)//3:2*len(params)//3]

            dist_x = m.shape[2] / (bspline_order + 1)

            for i in range(n_ctrl_points_x + bspline_order):
                y_params[i::n_ctrl_points_x + bspline_order] = self.random_state.rand() * 2 * dist_x * self.bspline_bend_lim - dist_x * self.bspline_bend_lim

            dists_y = np.zeros(n_ctrl_points_x + bspline_order - 1)
            thetas_y = np.zeros(n_ctrl_points_x + bspline_order - 1)
            for i in range(1, n_ctrl_points_x + bspline_order):
                dists_y[i-1] = y_params[i] - y_params[i-1]    
                thetas_y[i-1] = np.arctan2(dists_y[i-1], dist_x)

            max_y_dist = m.shape[1] / (bspline_order - 1)

            tot_displacement = np.sqrt(2) * max_y_dist * np.sqrt(1 - np.cos(thetas_y))
            x_displacement = tot_displacement * np.cos(thetas_y / 2)
            y_displacement = tot_displacement * np.sin(thetas_y / 2)

            # Initialize a variable to keep track of the cumulative adjustment in x
            cumulative_x_adjustment = 0

            # Adjust the x-coordinates of the control points considering the cumulative effect
            for i in range(1, n_ctrl_points_x + bspline_order):
                # Calculate the required x-distance to maintain the original distance between control points
                required_x_dist = dist_x - np.sqrt(dist_x**2 - dists_y[i-1]**2)

                # Calculate the adjustment needed for this control point
                x_adjustment = required_x_dist

                # Update the cumulative adjustment
                cumulative_x_adjustment += x_adjustment

                # Apply the cumulative adjustment to the current control point
                if i < n_ctrl_points_x + bspline_order:
                    x_params[i::n_ctrl_points_x + bspline_order] += cumulative_x_adjustment


            factor = np.linspace(-1,1,bspline_order + 1)
            for i in range(bspline_order + 1):
                for j in range(bspline_order + 1):
                    start_index = j * (n_ctrl_points_x + bspline_order) * (bspline_order + 1) + 1 + i * (n_ctrl_points_x + bspline_order)  # Start of the larger group
                    end_index = start_index + (n_ctrl_points_x + bspline_order) - 1 # End of the middle smaller group (5 elements)
                    # Modify the elements in this range
                    x_params[start_index:end_index] += x_displacement * factor[i]
                    y_params[start_index:end_index] += y_displacement * factor[i]


            x_params -= np.mean(x_params)
            y_params -= np.mean(y_params)

            params += self.random_state.randn(params.shape[0]) * sigma

            bspline_tfm.SetParameters(tuple(params))
            t = sitk.CompositeTransform(3)
            t.AddTransform(bspline_tfm)
            t.AddTransform(aff_tfm)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(raw_img)
        resampler.SetInterpolator(self.interpolator)
        resampler.SetDefaultPixelValue(cval)
        resampler.SetTransform(t)
        result = np.array(sitk.GetArrayFromImage(resampler.Execute(raw_img)))

        # If restore_rng_state is True, restore the saved state of the random number generator
        if restore_rng_state:
            self.random_state.set_state(rng_state)
        return result

    def __call__(self, m):
        # Check if the input image is 3D or 4D
        if m.ndim == 3:
            return self._apply_transformation(m)
        elif m.ndim == 4:
            # Apply the transformation independently to each channel
            # For all channels except the last one, restore the RNG state after processing
            channels = [self._apply_transformation(m[channel_idx], restore_rng_state=(channel_idx != m.shape[0]-1))
                        for channel_idx in range(m.shape[0])]

            return np.stack(channels)
        else:
            raise ValueError("Input should be either 3D or 4D.")

class GradualGaussianBlur:
    """
    Apply Gaussian blur to the image with a spatial gradient across the depth dimension.
    Blur intensity varies from 'blur_start' to 'blur_end'.
    """
    def __init__(self, random_state, max_blur_start=2.0, max_blur_end=2.0, num_blurs=5, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.max_blur_start = max_blur_start
        self.max_blur_end = max_blur_end
        self.num_blurs = num_blurs
        assert self.num_blurs > 1, "Number of blurs must be > 1."
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            return self.apply_gradual_blur(m)
        return m

    def apply_gradual_blur(self, data):
        if data.ndim != 4:  # Check if the input is 4D (CxDxHxW)
            raise ValueError("Input must be 4D (CxDxHxW).")

        assert self.num_blurs <= data.shape[1], "Number of blurs must be <= depth of the image."
        blurred_images = []
        blur_start = self.random_state.uniform(0, self.max_blur_start)
        blur_end = self.random_state.uniform(0, self.max_blur_end)
        sigma_values = np.linspace(blur_start, blur_end, self.num_blurs)
        for sigma in sigma_values:
            blurred_image = gaussian_filter(data, sigma=[0, sigma, sigma, sigma])
            blurred_images.append(blurred_image)

        return self.weighted_average(blurred_images, data.shape[1])

    def weighted_average(self, blurred_images, depth):
        new_image = np.zeros_like(blurred_images[0])
        depth_indices = np.linspace(0, depth, len(blurred_images))

        curr_idx = 1

        for d in range(depth):
            if d > depth_indices[curr_idx]:
                curr_idx += 1
            weight = (depth_indices[curr_idx] - d) / (depth_indices[curr_idx] - depth_indices[curr_idx - 1])
            new_image[:, d, :, :] = blurred_images[curr_idx][:, d, :, :] * (1 - weight) + blurred_images[curr_idx - 1][:, d, :, :] * weight

        return new_image


class BSplineDeformation:
    """ Apply B-Spline transformations to 3D patches. """
    def __init__(self, random_state, order=3, execution_probability=0.2, cval=None, spacing=100, sigma=20, use_z=True, interpolator='linear', **kwargs):
        self.random_state = random_state
        self.order = order
        self.execution_probability = execution_probability
        self.cval = cval
        self.spacing = spacing
        self.sigma = sigma
        self.use_z = use_z
        if interpolator == 'bspline':
            self.interpolator = sitk.sitkBSpline
        elif interpolator == 'linear':
            self.interpolator = sitk.sitkLinear
        elif interpolator == 'nn':
            self.interpolator = sitk.sitkNearestNeighbor
        else:
            raise NotImplementedError("ERROR: interpolator " + str(interpolator) + " not implemented.")


    def __call__(self, m):
        assert m.ndim == 3
        if self.random_state.uniform() < self.execution_probability:
            if self.cval is None:
                cval = np.median(m)
            else:
                cval = self.cval
            if self.use_z:
                bsp_grid_size = [math.floor(m.shape[2]/self.spacing), math.floor(m.shape[1]/self.spacing), math.floor(m.shape[0]/self.spacing)]
                raw_img = sitk.GetImageFromArray(m)
                t = sitk.BSplineTransformInitializer(raw_img, bsp_grid_size, order=3)
            else:
                bsp_grid_size = [math.floor(m.shape[2]/self.spacing), math.floor(m.shape[1]/self.spacing)]
                imgs = [sitk.GetImageFromArray(m[i], isVector=False) for i in range(m.shape[0])]
                t = sitk.BSplineTransformInitializer(imgs[0], bsp_grid_size, order=self.order)
            params = np.asarray(t.GetParameters(), dtype=np.float64)
            params = params + self.random_state.randn(params.shape[0]) * self.sigma
            t.SetParameters(tuple(params))
            resampler = sitk.ResampleImageFilter()
            if self.use_z:
                resampler.SetReferenceImage(raw_img)
            else:
                resampler.SetReferenceImage(imgs[0])
            resampler.SetInterpolator(self.interpolator)
            resampler.SetDefaultPixelValue(cval)
            resampler.SetTransform(t)
            if self.use_z:
                result = np.array(sitk.GetArrayFromImage(resampler.Execute(raw_img)))
            else:
                result = np.array([sitk.GetArrayFromImage(resampler.Execute(img)) for img in imgs])
            return result
        return m

# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=2000, sigma=50, execution_probability=0.1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m


def blur_boundary(boundary, sigma):
    boundary = gaussian(boundary, sigma=sigma)
    boundary[boundary >= 0.5] = 1
    boundary[boundary < 0.5] = 0
    return boundary


class CropToFixed:
    def __init__(self, random_state, size=(256, 256), centered=False, **kwargs):
        self.random_state = random_state
        self.crop_y, self.crop_x = size
        self.centered = centered

    def __call__(self, m):
        def _padding(pad_total):
            half_total = pad_total // 2
            return (half_total, pad_total - half_total)

        def _rand_range_and_pad(crop_size, max_size):
            """
            Returns a tuple:
                max_value (int) for the corner dimension. The corner dimension is chosen as `self.random_state(max_value)`
                pad (int): padding in both directions; if crop_size is lt max_size the pad is 0
            """
            if crop_size < max_size:
                return max_size - crop_size, (0, 0)
            else:
                return 1, _padding(crop_size - max_size)

        def _start_and_pad(crop_size, max_size):
            if crop_size < max_size:
                return (max_size - crop_size) // 2, (0, 0)
            else:
                return 0, _padding(crop_size - max_size)

        _, y, x = m.shape

        if not self.centered:
            y_range, y_pad = _rand_range_and_pad(self.crop_y, y)
            x_range, x_pad = _rand_range_and_pad(self.crop_x, x)

            y_start = self.random_state.randint(y_range)
            x_start = self.random_state.randint(x_range)

        else:
            y_start, y_pad = _start_and_pad(self.crop_y, y)
            x_start, x_pad = _start_and_pad(self.crop_x, x)

        result = m[:, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
        return np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect')


class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)  # Z
    ]

    def __init__(self, ignore_index=None, aggregate_affinities=False, append_label=False, **kwargs):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        :param aggregate_affinities: aggregate affinities with the same offset across Z,Y,X axes
        :param append_label: if True append the orignal ground truth labels to the last channel
        :param blur: Gaussian blur the boundaries
        :param sigma: standard deviation for Gaussian kernel
        """
        self.ignore_index = ignore_index
        self.aggregate_affinities = aggregate_affinities
        self.append_label = append_label

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()
        boundary_arr = [np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels]
        channels = np.stack(boundary_arr)
        results = []
        if self.aggregate_affinities:
            assert len(kernels) % 3 == 0, "Number of kernels must be divided by 3 (one kernel per offset per Z,Y,X axes"
            # aggregate affinities with the same offset
            for i in range(0, len(kernels), 3):
                # merge across X,Y,Z axes (logical OR)
                xyz_aggregated_affinities = np.logical_or.reduce(channels[i:i + 3, ...]).astype(np.int)
                # recover ignore index
                xyz_aggregated_affinities = _recover_ignore_index(xyz_aggregated_affinities, m, self.ignore_index)
                results.append(xyz_aggregated_affinities)
        else:
            results = [_recover_ignore_index(channels[i], m, self.ignore_index) for i in range(channels.shape[0])]

        if self.append_label:
            # append original input data
            results.append(m)

        # stack across channel dim
        return np.stack(results, axis=0)

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def get_kernels(self):
        raise NotImplementedError


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, blur=False, sigma=1, mode='thick', blobs=False, **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.blur = blur
        self.sigma = sigma
        self.mode = mode
        self.blobs = blobs

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        if self.blur:
            boundaries = blur_boundary(boundaries, self.sigma)

        results = []
        if self.blobs:
            blobs = (m > 0).astype('uint8')
            results.append(_recover_ignore_index(blobs, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class BlobsWithBoundary:
    def __init__(self, mode=None, append_label=False, blur=False, sigma=1, **kwargs):
        if mode is None:
            mode = ['thick', 'inner', 'outer']
        self.mode = mode
        self.append_label = append_label
        self.blur = blur
        self.sigma = sigma

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        results = [(m > 0).astype('uint8')]

        for bm in self.mode:
            boundary = find_boundaries(m, connectivity=2, mode=bm)
            if self.blur:
                boundary = blur_boundary(boundary, self.sigma)
            results.append(boundary)

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class BlobsToMask:
    """
    Returns binary mask from labeled image, i.e. every label greater than 0 is treated as foreground.

    """

    def __init__(self, append_label=False, boundary=False, cross_entropy=False, **kwargs):
        self.cross_entropy = cross_entropy
        self.boundary = boundary
        self.append_label = append_label

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        mask = (m > 0).astype('uint8')
        results = [mask]

        if self.boundary:
            outer = find_boundaries(m, connectivity=2, mode='outer')
            if self.cross_entropy:
                # boundary is class 2
                mask[outer > 0] = 2
                results = [mask]
            else:
                results.append(outer)

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class RandomLabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border. Then the offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the ground
    truth  (think of it as a boundary denoising scheme).
    """

    def __init__(self, random_state, max_offset=10, ignore_index=None, append_label=False, z_offset_scale=2, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label, aggregate_affinities=False)
        self.random_state = random_state
        self.offsets = tuple(range(1, max_offset + 1))
        self.z_offset_scale = z_offset_scale

    def get_kernels(self):
        rand_offset = self.random_state.choice(self.offsets)
        axis_ind = self.random_state.randint(3)
        # scale down z-affinities due to anisotropy
        if axis_ind == 2:
            rand_offset = max(1, rand_offset // self.z_offset_scale)

        rand_axis = self.AXES_TRANSPOSE[axis_ind]
        # return a single kernel
        return [self.create_kernel(rand_axis, rand_offset)]


class LabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels (which can be seen
    as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf)
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, aggregate_affinities=False, z_offsets=None,
                 **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label,
                         aggregate_affinities=aggregate_affinities)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), 'offsets must be a list or a tuple'
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"
        if z_offsets is not None:
            assert len(offsets) == len(z_offsets), 'z_offsets length must be the same as the length of offsets'
        else:
            # if z_offsets is None just use the offsets for z-affinities
            z_offsets = list(offsets)
        self.z_offsets = z_offsets

        self.kernels = []
        # create kernel for every axis-offset pair
        for xy_offset, z_offset in zip(offsets, z_offsets):
            for axis_ind, axis in enumerate(self.AXES_TRANSPOSE):
                final_offset = xy_offset
                if axis_ind == 2:
                    final_offset = z_offset
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, final_offset))

    def get_kernels(self):
        return self.kernels


class LabelToZAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels (which can be seen
    as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf)
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), 'offsets must be a list or a tuple'
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"

        self.kernels = []
        z_axis = self.AXES_TRANSPOSE[2]
        # create kernels
        for z_offset in offsets:
            self.kernels.append(self.create_kernel(z_axis, z_offset))

    def get_kernels(self):
        return self.kernels


class LabelToBoundaryAndAffinities:
    """
    Combines the StandardLabelToBoundary and LabelToAffinities in the hope
    that that training the network to predict both would improve the main task: boundary prediction.
    """

    def __init__(self, xy_offsets, z_offsets, append_label=False, blur=False, sigma=1, ignore_index=None, mode='thick',
                 blobs=False, **kwargs):
        # blur only StandardLabelToBoundary results; we don't want to blur the affinities
        self.l2b = StandardLabelToBoundary(blur=blur, sigma=sigma, ignore_index=ignore_index, mode=mode, blobs=blobs)
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        boundary = self.l2b(m)
        affinities = self.l2a(m)
        return np.concatenate((boundary, affinities), axis=0)


class FlyWingBoundary:
    """
    Use if the volume contains a single pixel boundaries between labels. Gives the single pixel boundary in the 1st
    channel and the 'thick' boundary in the 2nd channel and optional z-affinities
    """

    def __init__(self, append_label=False, thick_boundary=True, ignore_index=None, z_offsets=None, **kwargs):
        self.append_label = append_label
        self.thick_boundary = thick_boundary
        self.ignore_index = ignore_index
        self.lta = None
        if z_offsets is not None:
            self.lta = LabelToZAffinities(z_offsets, ignore_index=ignore_index)

    def __call__(self, m):
        boundary = (m == 0).astype('uint8')
        results = [boundary]

        if self.thick_boundary:
            t_boundary = find_boundaries(m, connectivity=1, mode='outer', background=0)
            results.append(t_boundary)

        if self.lta is not None:
            z_affs = self.lta(m)
            for z_aff in z_affs:
                results.append(z_aff)

        if self.ignore_index is not None:
            for b in results:
                b[m == self.ignore_index] = self.ignore_index

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class LabelToMaskAndAffinities:
    def __init__(self, xy_offsets, z_offsets, append_label=False, background=0, ignore_index=None, **kwargs):
        self.background = background
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        mask = m > self.background
        mask = np.expand_dims(mask.astype(np.uint8), axis=0)
        affinities = self.l2a(m)
        return np.concatenate((mask, affinities), axis=0)


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-6, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / np.clip(self.std, a_min=self.eps, a_max=None)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value, max_value, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, m):
        norm_0_1 = (m - self.min_value) / self.value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)


class AdditiveGaussianNoise:
    def __init__(self, random_state, scale=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise
        return m


class AdditivePoissonNoise:
    def __init__(self, random_state, lam=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Relabel:
    """
    Relabel a numpy array of labels into a consecutive numbers, e.g.
    [10,10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        _, unique_labels = np.unique(m, return_inverse=True)
        m = unique_labels.reshape(m.shape)
        return m


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m


def get_transformer(config, min_value, max_value, mean, std):
    base_config = {'min_value': min_value, 'max_value': max_value, 'mean': mean, 'std': std}
    return Transformer(config, base_config)


class Transformer:
    def __init__(self, phase_config, base_config):
        self.phase_config = phase_config
        self.config_base = base_config
        self.seed = np.random.randint(10000000)

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    def refresh_seed(self):
        self.seed = np.random.randint(10000000)

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('pytorch3dunet.augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
