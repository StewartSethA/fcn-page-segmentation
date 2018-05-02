import cv2
import numpy as np
from scipy.interpolate import griddata

def warp_images(img, img2, config={}, borderValue=(255,255,255), warp_amount=0.1):
    # TODO: Allow for random variation!!
    random_intervals = config.get("random_intervals", False)
    warp_variance = config.get("warp_variance", -1.0)
    warp_variance_max = config.get("warp_variance_max", warp_variance)
    if warp_variance_max > warp_variance:
        warp_variance = np.random.random() * (warp_variance_max-warp_variance) + warp_variance
    if warp_variance >= 0:
        w_mesh_variance = warp_variance
        h_mesh_variance = warp_variance
    else:
        w_mesh_variance = config.get("w_mesh_variance", 1.0) #max(0, self.w_mesh_variance + self.w_mesh_variance * (self.variable_warp_variance * (random.random()-0.5)) * w_mesh_interval / self.w_mesh_interval)
        h_mesh_variance = config.get("h_mesh_variance", 1.0) #max(0, self.h_mesh_variance + self.h_mesh_variance * (self.variable_warp_variance * (random.random()-0.5)) * h_mesh_interval / self.h_mesh_interval)
    warp_variance *= warp_amount
    mesh_interval = config.get("mesh_interval", -1)
    mesh_interval_max = config.get("mesh_interval_max", mesh_interval)
    if mesh_interval_max > mesh_interval:
        mesh_interval = np.random.random() * (mesh_interval_max-mesh_interval) + mesh_interval
    if mesh_interval > 0:
        w_mesh_interval = mesh_interval
        h_mesh_interval = mesh_interval
    else:
        w_mesh_interval = config.get("w_mesh_interval", 10) # max(1,self.w_mesh_interval + (self.variable_warp_spacing * (random.random()-0.5) * self.w_mesh_interval))
        h_mesh_interval = config.get("h_mesh_interval", 10) # max(1,self.h_mesh_interval + (self.variable_warp_spacing * (random.random()-0.5) * self.h_mesh_interval))

    width_center_variance = config.get('width_center_variance', 0.05)
    height_center_variance = config.get('height_center_variance', 0.05)

    #print("Warping by amounts ", h_mesh_variance, w_mesh_variance)

    h, w = img.shape[:2]
    ###### Make it so that it is an even interval close to the requested ######
    w_ratio = w / float(w_mesh_interval)
    h_ratio = h / float(h_mesh_interval)

    w_ratio = max(1, round(w_ratio))
    h_ratio = max(1, round(h_ratio))

    w_mesh_interval = w / w_ratio
    h_mesh_interval = h / h_ratio
    ###########################################################################

    w_scale_factor = (np.random.random() * 0.25) + 1.0 - 0.25 / 2.0
    w_scale_factor = min(1.0, w_scale_factor)
    c_i = w/2 - np.random.normal(0.25, height_center_variance) * w

    h_scale_factor = 1.0
    c_j = h/2 - np.random.normal(0.25, width_center_variance) * h

    new_width = int(w_scale_factor * w)
    grid_x, grid_y = np.mgrid[0:h, 0:new_width]

    source = []
    for i in np.arange(0, h+0.0001, h_mesh_interval):
        # print i
        for j in np.arange(0, w+0.0001, w_mesh_interval):
            source.append((i,j))

    destination = []
    for i, j in source:

        r_i, r_j = i, j

        r_i = np.random.normal(r_i, h_mesh_variance)
        r_i = h_scale_factor * r_i + c_i - h_scale_factor * c_i

        # Don't use the center because it makes sense that the word starts
        # at the beginning of the line
        r_j = np.random.normal(r_j, w_mesh_variance)
        r_j = w_scale_factor * r_j

        destination.append((r_i, r_j))

    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(h,new_width)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(h,new_width)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    # Smooth the displacement field with a Gaussian kernel
    warp_smoothing_variance = config.get("warp_smoothing_variance", 0.0)
    if warp_smoothing_variance > 0:
        kernel_dim = int(warp_smoothing_variance * 4.0 + 0.5)
        kernel = create_2d_gaussian(kernel_dim, warp_smoothing_variance)
        map_x_32 = convolve2d(map_x_32, kernel)
        map_y_32 = convolve2d(map_y_32, kernel)

    warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR, borderValue=borderValue)
    warped2 = cv2.remap(img2, map_x_32, map_y_32, cv2.INTER_LINEAR, borderValue=borderValue)

    width = img.shape[1]
    height = img.shape[0]
    pad_w = width-warped.shape[1]
    pad_h = height-warped.shape[0]
    pad_t = pad_h/2
    pad_b = pad_h - pad_t
    pad_l = pad_w / 2
    pad_r = pad_w - pad_l
    if len(warped.shape) == 3:
        warped = np.pad(warped, ((pad_t,pad_b), (pad_l, pad_r), (0,0)), mode='constant', constant_values=0)
    else:
        warped = np.pad(warped, ((pad_t,pad_b), (pad_l, pad_r)), mode='constant', constant_values=0)
    if len(warped2.shape) == 3:
        warped2 = np.pad(warped2, ((pad_t,pad_b), (pad_l, pad_r), (0,0)), mode='constant', constant_values=0)
    else:
        warped2 = np.pad(warped2, ((pad_t,pad_b), (pad_l, pad_r)), mode='constant', constant_values=0)
    return warped, warped2

def warp_image(img, borderValue=(255,255,255), config={}):
    # TODO: Allow for random variation!!
    random_intervals = config.get("random_intervals", False)
    warp_variance = config.get("warp_variance", -1.0)
    warp_variance_max = config.get("warp_variance_max", warp_variance)
    if warp_variance_max > warp_variance:
        warp_variance = np.random.random() * (warp_variance_max-warp_variance) + warp_variance
    if warp_variance >= 0:
        w_mesh_variance = warp_variance
        h_mesh_variance = warp_variance
    else:
        w_mesh_variance = config.get("w_mesh_variance", 1.0) #max(0, self.w_mesh_variance + self.w_mesh_variance * (self.variable_warp_variance * (random.random()-0.5)) * w_mesh_interval / self.w_mesh_interval)
        h_mesh_variance = config.get("h_mesh_variance", 1.0) #max(0, self.h_mesh_variance + self.h_mesh_variance * (self.variable_warp_variance * (random.random()-0.5)) * h_mesh_interval / self.h_mesh_interval)

    mesh_interval = config.get("mesh_interval", -1)
    mesh_interval_max = config.get("mesh_interval_max", mesh_interval)
    if mesh_interval_max > mesh_interval:
        mesh_interval = np.random.random() * (mesh_interval_max-mesh_interval) + mesh_interval
    if mesh_interval > 0:
        w_mesh_interval = mesh_interval
        h_mesh_interval = mesh_interval
    else:
        w_mesh_interval = config.get("w_mesh_interval", 10) # max(1,self.w_mesh_interval + (self.variable_warp_spacing * (random.random()-0.5) * self.w_mesh_interval))
        h_mesh_interval = config.get("h_mesh_interval", 10) # max(1,self.h_mesh_interval + (self.variable_warp_spacing * (random.random()-0.5) * self.h_mesh_interval))

    width_center_variance = config.get('width_center_variance', 0.05)
    height_center_variance = config.get('height_center_variance', 0.05)

    #print("Warping by amounts ", h_mesh_variance, w_mesh_variance)

    h, w = img.shape[:2]
    ###### Make it so that it is an even interval close to the requested ######
    w_ratio = w / float(w_mesh_interval)
    h_ratio = h / float(h_mesh_interval)

    w_ratio = max(1, round(w_ratio))
    h_ratio = max(1, round(h_ratio))

    w_mesh_interval = w / w_ratio
    h_mesh_interval = h / h_ratio
    ###########################################################################

    w_scale_factor = 1.0 #(np.random.random() * 0.25) + 1.0 - 0.25 / 2.0
    w_scale_factor = min(1.0, w_scale_factor)
    c_i = w/2 - np.random.normal(0.25, height_center_variance) * w

    h_scale_factor = 1.0
    c_j = h/2 - np.random.normal(0.25, width_center_variance) * h

    new_width = int(w_scale_factor * w)
    grid_x, grid_y = np.mgrid[0:h, 0:new_width]

    source = []
    for i in np.arange(0, h+0.0001, h_mesh_interval):
        # print i
        for j in np.arange(0, w+0.0001, w_mesh_interval):
            source.append((i,j))

    destination = []
    for i, j in source:

        r_i, r_j = i, j

        r_i = np.random.normal(r_i, h_mesh_variance)
        r_i = h_scale_factor * r_i + c_i - h_scale_factor * c_i

        # Don't use the center because it makes sense that the word starts
        # at the beginning of the line
        r_j = np.random.normal(r_j, w_mesh_variance)
        r_j = w_scale_factor * r_j

        destination.append((r_i, r_j))

    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(h,new_width)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(h,new_width)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    # Smooth the displacement field with a Gaussian kernel
    warp_smoothing_variance = config.get("warp_smoothing_variance", 0.0)
    if warp_smoothing_variance > 0:
        kernel_dim = int(warp_smoothing_variance * 4.0 + 0.5)
        kernel = create_2d_gaussian(kernel_dim, warp_smoothing_variance)
        map_x_32 = convolve2d(map_x_32, kernel)
        map_y_32 = convolve2d(map_y_32, kernel)

    warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR, borderValue=borderValue) #(255,255,255))
    width = img.shape[1]
    height = img.shape[0]
    pad_w = width-warped.shape[1]
    pad_h = height-warped.shape[0]
    pad_t = pad_h/2
    pad_b = pad_h - pad_t
    pad_l = pad_w / 2
    pad_r = pad_w - pad_l
    if len(img.shape) == 2:
        warped = np.pad(warped, ((pad_t,pad_b), (pad_l, pad_r)), mode='constant', constant_values=1.0)
    else:
        warped = np.pad(warped, ((pad_t,pad_b,0), (pad_l, pad_r,0)), mode='constant', constant_values=1.0)
    return warped
