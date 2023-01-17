import glob
import os

import geopandas as gpd
import laspy as lp
import numpy as np
import scipy.ndimage as ndimage
import tifffile as tiff
from patchify import patchify
from scipy import interpolate
from scipy.spatial import cKDTree as kdtree
from skimage.measure import regionprops
from skimage.segmentation import watershed
from sklearn.model_selection import train_test_split


def get_by_ext(src, ext=".tif"):
    if not ext.startswith("."):
        ext = "." + ext
    return sorted(glob.glob(os.path.join(src, f"*{ext}")))


def patch_train_label(
    rgb_fns,
    label_fns,
    img_size,
    channels=False,
    merge_channel=False,
    add_ndvi=False,
    squash=True,
    img_size_override=None,
):
    """
    Patchifies images and labels into square patches

    Args:
        raster (str): Location of rgb(i) images
        labels (str): Location of label images
        img_size (int): Patch size (single side only)
        channels (int, optional): Number of channels in the output patches. Defaults to
            False to use all available channels.
        merge_channel (str, optional): Location of images to be merged with raster as
            extra channel. Defaults to False.
        mask_class (bool, optional): Set to True to replace messed up "ignore" mask
            value with mask_val (this should fixed in actual image masking and removed). Defaults to False.
        mask_val (int, optional): Value of the desired "ignore" mask. Defaults to -1.

    Returns:
        data_train (ndarray): Array of observations
        data_label (ndarray): Array of binarized labels
        data_label_inst (ndarray): Array of labels with classes intact
    """

    assert len(rgb_fns) > 0, "Raster list is empty."
    samp_rast = tiff.imread(rgb_fns[0])
    img_base_size = img_size_override if img_size_override else samp_rast.shape[0]
    n = len(rgb_fns)
    m = (img_base_size // img_size) ** 2
    bad_ims = []

    if not channels:
        channels = samp_rast.shape[-1]

    # if merge_channel:
    #     channels += tiff.imread(merge_channel[0]).shape[-1]

    data_train = np.zeros((n * m, img_size, img_size, channels))
    data_label = np.zeros((n * m, img_size, img_size))

    for k in range(n):
        # Read in RGB and labels
        rgb = tiff.imread(rgb_fns[k])
        labels = tiff.imread(label_fns[k])

        if merge_channel:
            # Add NIR
            nir = tiff.imread(merge_channel[k])
            r = np.dstack((rgb, nir))

            if add_ndvi:
                # Add NDVI
                # Allow division by zero
                np.seterr(divide="ignore", invalid="ignore")
                r = r.astype(np.float64)
                ndvi = (r[..., -1] - r[..., 0]) / (r[..., -1] + r[..., 0])
                r = r.astype(np.uint16)
                ndvi[np.isnan(ndvi)] = -1
                # Normalize and convert to 255 range
                ndvi = (((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())) * 255).astype(
                    np.uint16
                )
                r = np.dstack((r, ndvi))
        else:
            # Just use RGB
            r = rgb

        if squash:
            # Squash all class labels into single class
            labels[labels > 0] = 1

        if (
            img_size_override
            and (r.shape[0] == img_size_override and r.shape[1] == img_size_override)
        ) or not img_size_override:
            # Only read in the specified number of channels from input raster
            patches_train = patchify(
                r,
                (img_size, img_size, channels),
                step=img_size,
            )
            patches_label = patchify(labels, (img_size, img_size), step=img_size)
            # print(patches_label.shape)
            data_train[k * m : (k + 1) * m, :, :, :] = patches_train.reshape(
                -1, img_size, img_size, channels
            )
            data_label[k * m : (k + 1) * m, :, :] = patches_label.reshape(
                -1, img_size, img_size
            )
        else:
            bad_ims.append(rgb_fns[k])

    if img_size_override and len(bad_ims) > 0:
        print("bad ims length:", len(bad_ims))
        print("m:", m)
        print("Cull rows:", -len(bad_ims) * m)
        print("Shape before:", data_train.shape)
        bad_im_idx = np.arange(-len(bad_ims) * m, 0, 1)
        data_train = np.delete(data_train, bad_im_idx, axis=0)
        data_label = np.delete(data_label, bad_im_idx, axis=0)
        print("Shape after:", data_train.shape)

    data_label = np.expand_dims(data_label, axis=-1)
    data_train = data_train.astype("float") / 255

    print(
        f"\nPatched data sizes:\ndata_train: {data_train.shape}\ndata_label: {data_label.shape}"
    )

    return data_train, data_label


def prep_data(
    data_dir,
    seed=157,
    include_nir=True,
    add_ndvi=True,
    squash=True,
    patch_size=512,
    img_size_override=None,
):
    # DATASET
    rgb_fns = []
    nir_fns = [] if include_nir else None
    label_fns = []

    dirs = glob.glob(f"{data_dir}/*")
    print("Number of directories:", len(dirs))

    for dir in dirs:
        rgb = glob.glob(f"{dir}/rgb/*.tif")[0]
        nir = glob.glob(f"{dir}/nir/*.tif")[0] if include_nir else None
        labels = glob.glob(f"{dir}/labels/*.tif")[0]
        rgb_fns.append(rgb)
        if include_nir:
            nir_fns.append(nir)
        label_fns.append(labels)

    assert len(rgb_fns) == len(label_fns), "RGB and Labels file counts do not match."
    # print("RGB size:", len(rgb_fns))
    # if include_nir:
    #     print("NIR size:", len(nir_fns))
    # print("Labels size:", len(label_fns))

    if include_nir and add_ndvi:
        data_train, data_label = patch_train_label(
            rgb_fns,
            label_fns,
            patch_size,
            channels=5,
            merge_channel=nir_fns,
            add_ndvi=True,
            squash=squash,
            img_size_override=img_size_override,
        )
    elif include_nir:
        data_train, data_label = patch_train_label(
            rgb_fns,
            label_fns,
            patch_size,
            channels=4,
            merge_channel=nir_fns,
            squash=squash,
            img_size_override=img_size_override,
        )
    else:
        data_train, data_label = patch_train_label(
            rgb_fns,
            label_fns,
            patch_size,
            channels=3,
            squash=squash,
            img_size_override=img_size_override,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        data_train,
        data_label,
        test_size=0.33,
        shuffle=True,
        random_state=seed,
    )

    print(
        f"\nDataset sizes:\n\
    X_train: {X_train.shape}\n\
    y_train: {y_train.shape}\n\
    X_test: {X_test.shape}\n\
    y_test: {y_test.shape}"
    )

    # # Patchify watershed data (pre-patchified)
    # WS_DIR = os.path.join(data_dir, "watershed")
    # WS_RGBI_DIR = os.path.join(WS_DIR, f"rgbi/{ws_rgb}/512")
    # WS_LABEL_DIR = os.path.join(WS_DIR, f"labels/{ws_label}/512")

    # data_train, data_label, data_label_inst = patch_train_label(
    #     WS_RGBI_DIR, WS_LABEL_DIR, 128, channels=channels, mask_class=False
    # )

    # # Always use the hand-labeled test split as final test (outside KF CV) because
    # # we know it is higher quality
    # X_train = np.concatenate((X_train, data_train), axis=0)
    # y_train = np.concatenate((y_train, data_label), axis=0)
    # inst_train = np.concatenate((inst_train, data_label_inst))

    # pct_bg = (np.count_nonzero(y_train == 0) / len(y_train.ravel())) * 100
    # pct_trees = (np.count_nonzero(y_train == 1) / len(y_train.ravel())) * 100
    # pct_masked = (np.count_nonzero(y_train == -1) / len(y_train.ravel())) * 100
    # print("\nWatershed percents")
    # print("--------------------")
    # print(f"% BG: {pct_bg:.2f}%")
    # print(f"% Trees: {pct_trees:.2f}%")
    # print(f"% Masked: {pct_masked:.2f}%")

    # print(
    #     f"\nSizes after adding watershed data:\n\
    # X_train: {X_train.shape}\n\
    # y_train: {y_train.shape}\n\
    # X_test: {X_test.shape}\n\
    # y_test: {y_test.shape}"
    # )

    return X_train, y_train, X_test, y_test


def filter_labels(labels, img, band, area, ecc, ar, abr, intensity):
    """Takes a set of labels and returns a filtered set based on regionprops parameters.

    Args:
        labels (ndarray): Watershed labels
        img (ndarray): Image to pair with labels for regionprops extraction
        band (int): The band to use for intensity in regionprops
        area (float): Minimum area of filtered regions
        ecc (float): Maximum eccentricity of filtered regions
        ar (float): Minimum ratio of minor and major axes (1=square, 0.5=rectangle)
        abr (float): Minimum ratio of area of non-zero pixels compared to bounding box area
        intensity (float): Minimum band intensity of corresponding image

    Returns:
        filtered_labels (ndarray): Array of the filtered labels
        bbox (list): List of the coordinates of each label's bounding box
    """
    if band:
        regions = regionprops(labels, img[..., band])
    else:
        regions = regionprops(labels, img)
    filtered_labels = np.zeros((labels.shape[0], labels.shape[1]), dtype=int)
    bbox = []
    for region in regions:
        if (
            region.area >= area
            and (region.axis_minor_length / region.axis_major_length >= ar)
            and (region.eccentricity <= ecc)
            and (region.area / region.area_bbox >= abr)
            and (region.intensity_mean >= intensity)
        ):
            filtered_labels[region.coords[:, 0], region.coords[:, 1]] = region.label
            minr, minc, maxr, maxc = region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            bbox.append([bx, by])

    return filtered_labels, bbox


def watershed_labels(img, neighborhood_size, threshold, min_height):
    p_smooth = ndimage.gaussian_filter(img, threshold)
    p_max = ndimage.maximum_filter(p_smooth, neighborhood_size)
    local_maxima = p_smooth == p_max
    local_maxima[img == 0] = 0
    labeled, num_objects = ndimage.label(local_maxima)
    xy = np.array(
        ndimage.center_of_mass(
            input=img, labels=labeled, index=range(1, num_objects + 1)
        )
    )
    binary_mask = np.where(img >= min_height, 1, 0)
    binary_mask = ndimage.binary_fill_holes(binary_mask).astype(int)

    labels = watershed(-img, labeled, mask=binary_mask)
    return labels, xy


def unpickle(pickle):
    a = np.zeros((len(pickle), len(pickle[0])))
    for i in range(len(pickle)):
        a[i, :] = pickle[i]
    return a


def las2chm(las_file):
    las = lp.read(las_file)
    points = las.xyz.copy()
    return_num = las.return_number.copy()
    num_of_returns = las.number_of_returns.copy()
    classification = las.classification.copy()
    select = classification != 5
    select += (return_num == 1) * (num_of_returns == 1)
    select += (return_num == 2) * (num_of_returns == 2)
    select += (return_num == 3) * (num_of_returns == 3)
    select += (return_num == 4) * (num_of_returns == 4)
    select += (return_num == 5) * (num_of_returns == 5)
    points = points[~select]
    tr = kdtree(points)
    distances, indices = tr.query(points, k=25, workers=-1)
    distances = distances[:, -1]
    thr = 2.0
    select = distances > thr
    points = points[~select]
    orginal_points = las.xyz.copy()
    tr = kdtree(orginal_points)
    distances, indices = tr.query(points, k=10, workers=-1)
    distances = distances[:, -1]
    indices = np.unique(indices[distances < 0.5])
    points = np.vstack((points, orginal_points[indices]))
    slice_position = np.mean(points[:, 1])
    width = 5
    slice_org = np.sqrt((orginal_points[:, 1] - slice_position) ** 2) <= width
    slice = np.sqrt((points[:, 1] - slice_position) ** 2) <= width
    gridsize = 1.0  # [m]
    ground_points = las.xyz[las.classification == 2]
    grid_x = ((ground_points[:, 0] - ground_points[:, 0].min()) / gridsize).astype(
        "int"
    )
    grid_y = ((ground_points[:, 1] - ground_points[:, 1].min()) / gridsize).astype(
        "int"
    )
    grid_index = grid_x + grid_y * grid_x.max()
    df = gpd.GeoDataFrame(
        {"gi": grid_index, "gx": grid_x, "gy": grid_y, "height": ground_points[:, 2]}
    )
    df2 = df.sort_values(["gx", "gy", "height"], ascending=[True, True, True])
    df3 = df2.groupby("gi")[["gx", "gy", "height"]].last()
    grid_x = np.array(df3["gx"])
    grid_y = np.array(df3["gy"])
    max_height = np.array(df3["height"])
    DTM = np.ones((grid_x.max() + 1, grid_y.max() + 1)) * np.nan
    DTM[grid_x, grid_y] = max_height
    mask = np.isnan(DTM)
    xx, yy = np.meshgrid(np.arange(DTM.shape[0]), np.arange(DTM.shape[1]))
    valid_x = xx[~mask]
    valid_y = yy[~mask]
    newarr = DTM[~mask]
    DTM_interp = interpolate.griddata(
        (valid_x, valid_y), newarr.ravel(), (xx, yy), method="linear"
    )
    gridsize = 1.0  # [m]
    filt_points = points
    grid_x = ((filt_points[:, 0] - filt_points[:, 0].min()) / gridsize).astype("int")
    grid_y = ((filt_points[:, 1] - filt_points[:, 1].min()) / gridsize).astype("int")
    grid_index = grid_x + grid_y * grid_x.max()
    df = gpd.GeoDataFrame(
        {"gi": grid_index, "gx": grid_x, "gy": grid_y, "height": filt_points[:, 2]}
    )
    df2 = df.sort_values(["gx", "gy", "height"], ascending=[True, True, True])
    df3 = df2.groupby("gi")[["gx", "gy", "height"]].last()
    grid_x = np.array(df3["gx"])
    grid_y = np.array(df3["gy"])
    max_height = np.array(df3["height"])
    DSM = np.ones((grid_x.max() + 1, grid_y.max() + 1)) * np.nan
    DSM[grid_x, grid_y] = max_height
    CHM = DSM - DTM_interp
    CHM[np.isnan(CHM)] = 0
    return CHM
