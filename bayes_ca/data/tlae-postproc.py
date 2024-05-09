from pathlib import Path

import click
import numpy as np
import nibabel as nib
from scipy import stats
from sklearn import linear_model
from nilearn import image, masking


def _regress_confounds(data, confounds):
    """
    Regress out confounds from data.
    """
    lr = linear_model.LinearRegression()
    lr.fit(confounds, data.T)
    regr_data = data - np.dot(lr.coef_, confounds.T) - lr.intercept_[:, None]
    # Note some % of values on cortical surface are NaNs,
    # so the following will throw an error
    zscore_data = stats.zscore(regr_data, axis=1)
    return np.nan_to_num(zscore_data)


def create_sessions(postproc_dir, subject_id):
    """
    Row stack individual fMRI files to yield one time x voxel matrix per session
    """
    p = Path(postproc_dir)
    for i in range(1, 6):
        ses_id = f"ses-wk{i}"
        files = list(p.glob(f"{subject_id}_{ses_id}*.npy"))
        files = sorted(files)
        ses_data = np.row_stack([np.load(f) for f in files])
        np.save(f"{subject_id}_{ses_id}_AG_roi", ses_data)
    return


@click.command()
@click.option("--datadir")
@click.option("--outdir")
@click.option("--subject")
def main(datadir, outdir, subject):
    """
    Known issues identified in Lee et al., 2024, NeurIPS :
        - sub-s103, no ses-wk3
        - after sub-s106 w4recap, w5recap was skipped
        - sub-s112, no ses-wk6 placement
        - sub-s201, no ses-wk6 placement
    """
    s = Path(datadir, subject)

    vols = sorted(s.rglob("*preproc_bold.nii.gz"))
    for vol in vols:
        sub, ses, task, space, res, _, _ = vol.name.split("_")
        print(f"Processing : {vol}")
        mask = s.rglob(f"**/func/*{ses}_{task}_{space}_{res}_desc-brain_mask.nii.gz")
        surfs = s.rglob(f"{sub}_{ses}_{task}*bold.func.gii")
        cfile = s.rglob(f"{sub}_{ses}_{task}_desc-confounds_timeseries.tsv")

        # load confounds
        conf = np.genfromtxt(next(cfile).as_posix(), names=True)
        conf_keys = [
            "trans_x",  # Motion and motion derivatives
            "trans_x_derivative1",
            "trans_y",
            "trans_y_derivative1",
            "trans_z",
            "trans_z_derivative1",
            "rot_x",
            "rot_x_derivative1",
            "rot_y",
            "rot_y_derivative1",
            "rot_z",
            "rot_z_derivative1",
            "framewise_displacement",
        ]
        conf_keys += [f"a_comp_cor_{i:02d}" for i in range(6)]
        conf_keys += ["cosine00", "cosine01"]

        try:
            regrs = np.column_stack([conf[c] for c in conf_keys])
        except ValueError:  # scan too short ; no cosine01 generated
            conf_keys = conf_keys[:-1]
            regrs = np.column_stack([conf[c] for c in conf_keys])
        regrs = np.nan_to_num(regrs)

        # clean data using linear regression of confounds
        mask = next(mask)
        masked_vol = masking.apply_mask(vol, mask)
        clean_vol = _regress_confounds(masked_vol.T, regrs)

        for surf in surfs:
            surf_data = np.column_stack([x.data for x in nib.load(surf).darrays])
            if "hemi-L" in str(surf):
                clean_surf_l = _regress_confounds(surf_data, regrs)
            elif "hemi-R" in str(surf):
                clean_surf_r = _regress_confounds(surf_data, regrs)

        # we'll also extract data for ang. gyr. ROI in volumetric space
        roi = Path(datadir, "..", "AG_mask.nii.gz")
        unmask_vol = masking.unmask(clean_vol.T, mask)
        # paper reports 3mm isotropic voxels, but 2mm acquired so resample
        rs_vol = image.resample_img(unmask_vol, np.eye(3) * 3)
        rs_mask = image.resample_to_img(roi, rs_vol, interpolation="nearest")
        ang_roi = masking.apply_mask(rs_vol, rs_mask)

        # Save hdf5 filem with all fields as n_samples x n_features
        # savepath = Path(outdir, f"{sub}_{ses}_{task}.h5")
        # with h5py.File(savepath, "w") as hf:
        #     grp = hf.create_group(ses)
        #     grp.create_dataset("surf-l", data=clean_surf_l.T)
        #     grp.create_dataset("surf-r", data=clean_surf_r.T)
        #     grp.create_dataset("vol", data=clean_vol.T)
        #     grp.create_dataset("regrs", data=regrs)
        #     # grp.create_dataset("ag-roi", data=ang_roi)

        np.save(Path(outdir, f"{sub}_{ses}_{task}_AG_roi"), ang_roi)

        # Inefficient but safer : re-load individual task files and combine
        # into larger session-specific time x voxel matrices
        create_sessions(outdir, subject)


if __name__ == "__main__":
    main()
