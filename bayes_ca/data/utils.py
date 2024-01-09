import os
import h5py
import requests
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image
from nilearn.maskers import NiftiMasker
from sklearn.linear_model import LinearRegression

_package_directory = os.path.dirname(os.path.abspath(__file__))
ANG_ROI_FILE_PATH = os.path.join(_package_directory, "AG_mask.nii.gz")


def _fetch_sherlock_data(data_dir=None):
    """
    Fetch sherlock dataset used in BrainIAK tutorials.
    By default, store in current working directory, unless
    data_dir is passed.

    Parameters
    ----------
    data_dir : str
        Path on disk to store sherlock dataset.
    """
    if data_dir is not None:
        os.chdir(data_dir)
    url = "https://ndownloader.figshare.com/files/9017983"
    resp = requests.get(url, allow_redirects=True)
    open("sherlock.h5", "wb").write(resp.content)
    with h5py.File("sherlock.h5", "r") as f:
        bold = f["BOLD"][()]
    return bold


def _resample_ang_roi(scan, mask):
    """

    Parameters
    ----------
    scan: niimg_like
        Reference Niimg for resampling.
    mask: str
        The (brain) mask within which to process data.
    """
    aff_orig = nib.load(scan).affine[:, -1]
    target_affine = np.column_stack([np.eye(4, 3) * 3, aff_orig])

    resampled_mask = image.resample_img(
        img=mask, target_affine=target_affine, interpolation="nearest"
    )

    return resampled_mask


def naturalistic_data(data_dir, gsr=False):
    """
    Fetch sherlock dataset used in OHBM Naturalistic Data
    tutorials. By default, store in current working directory,
    unless data_dir is passed.

    Parameters
    ----------
    data_dir : str
        Path on disk to store sherlock dataset.
    gsr : bool
        Whether or not to apply global-signal regression to the
        pre-parcellated data. Default False.

    Returns
    -------
    bold : np.arr
        Array of shape (n_regions, n_TRs, n_subjects)
    """
    bold = np.load(Path(data_dir, "Sherlock_AG_movie.npy")).T

    if gsr:
        global_signal = np.mean(bold, axis=0)
        pred_bold_gs = []

        for s in range(bold.shape[-1]):
            ols = LinearRegression(fit_intercept=False).fit(
                global_signal.T[s].reshape(-1, 1), bold[..., s].T
            )
            pred_bold_gs.append(ols.predict(global_signal.T[s].reshape(-1, 1)).T)

        gsr_bold = bold - np.stack(pred_bold_gs, axis=2)
        return gsr_bold

    return bold


def bbt_data(data_dir, gsr=False):
    """
    Assuming the preprocessed BBT (Du Bon, de la Brute et du Truand)
    data exists on disk, fetch and mask that data with predetermined
    regions-of-interest as defined in Baldassano et al. (2017).

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Location of the BBT pre-processed dataset on disk.
    gsr : bool
        Whether or not to apply global-signal regression to the
        pre-parcellated data. Default False.

    Returns
    -------
    bold : np.arr
        Array of shape (n_regions, n_TRs, n_subjs)

    """
    scans = []
    pattern = f"*task-GoodBadEvil_desc-fwhm5_bold.nii.gz"
    path = Path(data_dir)
    scans.append(sorted(path.rglob(pattern)))

    res_mask = _resample_ang_roi(scans[0][0], ANG_ROI_FILE_PATH)

    masker = NiftiMasker(
        mask_img=res_mask,
        standardize=False,
        memory=".nilearn-cache",
        memory_level=1,
        verbose=2,
    ).fit()

    ang_vx = [masker.transform(s).T for s in sum(scans, [])]
    bold = np.asarray([(v.T + np.random.random(v.T.shape)) for v in ang_vx]).T

    if gsr:
        global_signal = np.mean(bold, axis=0)
        pred_bold_gs = []

        for s in range(bold.shape[-1]):
            ols = LinearRegression(fit_intercept=False).fit(
                global_signal.T[s].reshape(-1, 1), bold[..., s].T
            )
            pred_bold_gs.append(ols.predict(global_signal.T[s].reshape(-1, 1)).T)

        gsr_bold = bold - np.stack(pred_bold_gs, axis=2)
        gsr_bold = gsr_bold + np.random.random(gsr_bold.shape)
        return gsr_bold

    return bold


def sherlock_data(gsr=False):
    """
    Replicating BrainIAK HMM tutorial on sherlock data
    (as distributed by the BrainIAK authors).
    For the full tutorial, please see :
    https://brainiak.org/tutorials/12-hmm/
    Here, we only replicate the event boundary estimation
    and compairison with a permuted null model.

    Parameters
    ----------
    gsr : bool
        Whether or not to apply global-signal regression to the
        pre-parcellated data.

    Returns
    -------
    BOLD : np.arr
        Array of shape (141, 1976, 17)
    """
    bold = _fetch_sherlock_data()

    if gsr:
        global_signal = np.mean(bold, axis=0)
        pred_bold_gs = []

        for s in range(bold.shape[-1]):
            ols = LinearRegression(fit_intercept=False).fit(
                global_signal.T[s].reshape(-1, 1), bold[..., s].T
            )
            pred_bold_gs.append(ols.predict(global_signal.T[s].reshape(-1, 1)).T)

        gsr_bold = bold - np.stack(pred_bold_gs, axis=2)
        return gsr_bold

    return bold
