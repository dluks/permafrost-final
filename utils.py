import glob
import os

import numpy as np
import tifffile as tiff
from patchify import patchify
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
    select_labels=False,
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
                r = r.astype(np.uint8)
                ndvi[np.isnan(ndvi)] = -1
                # Normalize and convert to 255 range
                ndvi = (((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())) * 255).astype(
                    np.uint8
                )
                r = np.dstack((r, ndvi))
        else:
            # Just use RGB
            r = rgb
        

        # Only include patches that are square and match the img_size_override
        # dimension (if set), and that contain no 0 (nodata) values.
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
            data_train[k * m : (k + 1) * m, :, :, :] = patches_train.reshape(
                -1, img_size, img_size, channels
            )
            data_label[k * m : (k + 1) * m, :, :] = patches_label.reshape(
                -1, img_size, img_size
            )
        else:
            bad_ims.append(rgb_fns[k])

    if img_size_override and len(bad_ims) > 0:
        print("\nRemoving irregular-shaped images...")
        print("-------------------------------------")
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

    return data_train, data_label


def prep_data(
    data_dir,
    seed=157,
    include_nir=True,
    add_ndvi=True,
    select_labels=False,
    squash=True,
    patch_size=512,
    img_size_override=None,
):
    # DATASET
    rgb_fns = []
    nir_fns = [] if include_nir else None
    label_fns = []

    dirs = glob.glob(f"{data_dir}/*")

    for dir in dirs:
        rgb = glob.glob(f"{dir}/rgb/*.tif")[0]
        nir = glob.glob(f"{dir}/nir/*.tif")[0] if include_nir else None
        labels = glob.glob(f"{dir}/labels/*.tif")[0]
        rgb_fns.append(rgb)
        if include_nir:
            nir_fns.append(nir)
        label_fns.append(labels)

    assert len(rgb_fns) == len(label_fns), "RGB and Labels file counts do not match."

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

    # Identify patches with nodata (0) values and cull
    print("\nRemoving images with nodata values...")
    print("-------------------------------------")
    print("Training set size with nodata:", data_train.shape[0])
    nodata_patches = []
    for patch in data_train[..., 0:3]:
        nodata_patches.append(len(patch[patch==0]) != 0)
    nodata_patches = np.asarray(nodata_patches)
    data_train = data_train[~nodata_patches]
    data_label = data_label[~nodata_patches]
    print(
        f"\nPatched data sizes w/o nodata:\ndata_train: {data_train.shape}\ndata_label: {data_label.shape}"
    )
    
    
    # Generate list of whether or not patches have labels
    print("\nRemoving images with no positive labels...")
    print("-------------------------------------")
    no_label_patches = []
    for patch in data_label:
        no_label_patches.append(np.all(patch == 0))

    no_label_patches = np.asarray(no_label_patches) # All true have no labels

    # Change the value of half of the patches that don't have labels (true -> false)
    no_label_patches_idx = np.where(no_label_patches)[0] # Don't have labels
    # print(no_label_patches_idx)
    print("Number of no label patches before removal:", no_label_patches_idx.shape[0])
    keep_n = round(no_label_patches_idx.shape[0] * 0.5) # 50% of count of no-label patches
    print("Keep n labels:", keep_n)
    keep_idx = np.sort(np.random.choice(no_label_patches_idx, keep_n, replace=False))
    print("# of keep idx:", len(keep_idx))
    # print("keep IDs:", keep_idx)

    no_label_patches[keep_idx] = False
    no_label_patches_idx = np.where(no_label_patches)[0]
    print("Number of no label patches after removal:", no_label_patches_idx.shape[0])

    data_label = data_label[~no_label_patches]
    data_train = data_train[~no_label_patches]
    
    if select_labels or squash:
        for patch in data_label:
            if select_labels:
                patch[~np.isin(patch, select_labels)] = 0
                
            if squash:
                # Squash all class labels into single class
                patch[patch > 0] = 1
    
    
    
    # Remove percentage of patches that have no positive label ids
    # no_label_patches = []
    # for patch in data_label:
    #     no_label_patches.append(np.all(patch == 0))

    X_train, X_test, y_train, y_test = train_test_split(
        data_train,
        data_label,
        test_size=0.33,
        shuffle=True,
        # random_state=seed,
    )

    print(
        f"\nDataset sizes:\n\
    X_train: {X_train.shape}\n\
    y_train: {y_train.shape}\n\
    X_test: {X_test.shape}\n\
    y_test: {y_test.shape}"
    )

    return X_train, y_train, X_test, y_test


def unpickle(pickle):
    a = np.zeros((len(pickle), len(pickle[0])))
    for i in range(len(pickle)):
        a[i, :] = pickle[i]
    return a
