import re
import glob
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image, maskers


def _get_segment(files):
    """
    Gets the segment of the movie (as encoded in the task
    value) for sorting. Since some segments were shown out
    of order, this is more reliable than a naive sort of
    the filenames.

    Parameters
    ----------
    files : list of str
        The list of filenames that should be sorted.

    task : str
        The name of the task, not encoding the segment.
        For example, if the task value is 'figures04'
        this string should be 'figures'.

    Returns
    -------
    segments: list of str
        The list of filenames, sorted by movie segment.
    """
    segments = sorted(files, key=lambda x: int(re.search(f"run-(\d+)", x).group(1)))
    return segments


def _nifti_mask_movie(scan, mask, confounds, smoothing_fwhm=None):
    """
    Cleans movie data, including standardizing and high-pass
    filtering at 0.007Hz. Corrects for supplied motion confounds
    and calculates and corrects for top 5 tCompCor confounds.
    Optionally smooths time series.

    Parameters
    ----------
    scan: niimg_like
        An in-memory niimg
    mask: str or Niimg-like
        The (brain) mask within which to process data.
    confounds: np.ndarray
        Any confounds to correct for in the cleaned data set.
    smoothing_fwhm : int
        (Optional) The size of the Gaussian smoothing kernel to apply.
    """
    # niftimask and clean data
    masker = maskers.NiftiMasker(
        mask_img=mask,
        t_r=1.49,
        standardize=True,
        detrend=False,
        high_pass=0.00714,
        smoothing_fwhm=smoothing_fwhm,
        memory=".nilearn-cache",
        memory_level=1,
        verbose=2,
    )
    compcor = image.high_variance_confounds(
        scan, mask_img=mask, n_confounds=5, percentile=5.0
    )
    confounds = np.hstack((confounds, compcor))
    cleaned = masker.fit_transform(scan, confounds=confounds)
    return masker.inverse_transform(cleaned)


def _resample_movie(scan, mask):
    """

    Parameters
    ----------
    scan: niimg_like
        An in-memory niimg
    mask: str
        The (brain) mask within which to process data.
    """
    if mask is not None:
        aff_orig = nib.load(mask).affine[:, -1]
    else:
        aff_orig = nib.load(scan).affine[:, -1]
    target_affine = np.column_stack([np.eye(4, 3) * 3, aff_orig])

    # remove NaNs and resample to 3mm isotropic
    nonan_scan = image.clean_img(
        scan, detrend=False, standardize=False, ensure_finite=True
    )
    resampled_scan = image.resample_img(
        img=nonan_scan, target_affine=target_affine, target_shape=(65, 77, 65)
    )
    resampled_mask = image.resample_img(
        img=mask, target_affine=target_affine, interpolation="nearest"
    )

    return resampled_scan, resampled_mask


def subset_and_process_bbt(bold_files, bold_txts, subject, n_segments=None, fwhm=None):
    """
    Note that the bbt is long,with 21 separate acquisitions.
    This function is therefore also designed to subset the movie
    to a set number of segments; for example, to match the number
    of frames across tasks.

    Although this behavior is off by default  (and controllable
    with the n_segments argument), note that if you choose to
    process the whole movie you can expect a very high memory
    usage.

    Parameters
    ----------
    bold_files : list
        A list of the BOLD file names to subset and process.
    bold_txts : list
        A list of motion regressors accompanying each BOLD file.
    subject : str
        Subject identifier string.
    n_segments : int
        The number of segments to subset from the movie.
        Will error if the number of segments requested is
        more than are available in the movie.
    fwhm : int
        The size of the Gaussian smoothing kernel to apply.

    Returns
    -------
    postproc_fname : str
        Filename for the concatenated postprocessed file,
        optionally smoothed to supplied FWHM.
    """
    if n_segments > 21:
        warnings.warn(
            "Too many segments requested ! Only 21 available. "
            "To return all segments, pass None."
        )
        return
    movie_segments = bold_files[:n_segments]
    regressors = bold_txts[:n_segments]

    # use the brain mask directly from templateflow,
    # so all subjects have the same number of voxels.
    tpl_mask = "./tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"
    postproc_fname = Path(
        f"{subject}", f"{subject}_task-GoodBadEvil_desc-fwhm{fwhm}_bold.nii.gz"
    )

    if postproc_fname.exists():
        print(f"File {postproc_fname} already exists; skipping")
        return postproc_fname

    postproc_segments = []
    for r, m in zip(regressors, movie_segments):
        res_scan, res_mask = _resample_movie(scan=m, mask=tpl_mask)
        confounds = np.loadtxt(r)
        postproc = _nifti_mask_movie(
            scan=res_scan, mask=res_mask, confounds=confounds, smoothing_fwhm=fwhm
        )
        postproc_segments.append(postproc)

    movie = image.concat_imgs(postproc_segments)
    nib.save(movie, postproc_fname)
    return postproc_fname


if __name__ == "__main__":
    # Assumes this script is co-located with data
    subjects = sorted(glob.glob("sub-*"))
    for s in subjects:
        files = Path(s).rglob(f"wrdc{s}*GoodBadEvil*.nii.gz")
        regressors = Path(s).rglob(f"rp_dc{s}*GoodBadEvil*bold.txt")
        segments = _get_segment([str(f) for f in files])
        matched_reg = _get_segment(str(r) for r in regressors)
        subset_and_process_bbt(segments, matched_reg, s, n_segments=8, fwhm=5)
