#!/bin/bash
#
#SBATCH --job-name=tlae_fMRIPrep
#SBATCH --output=tlae_fmriprep.%j.out
#SBATCH --time=1-00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-25
#SBATCH -p russpold,owners

# Define directories

DATADIR=$OAK/users/emdupre/think-like-an-expert/ds003233
OUTDIR=$SCRATCH/think-like-an-expert
SIFDIR=$OAK/users/emdupre/think-like-an-expert/
LICENSE=$HOME/submission_scripts

# Begin work section
subj_list=(`find $DATADIR -maxdepth 1 -type d -name 'sub-s*' -printf '%f\n' | sort -n -ts -k2.1`)
sub="${subj_list[$SLURM_ARRAY_TASK_ID]}"
echo "SUBJECT_ID: " $sub

singularity run --cleanenv -B ${DATADIR}:/data:ro \
	-B ${OUTDIR}:/out \
	-B ${LICENSE}/license.txt:/license/license.txt:ro \
	${SIFDIR}/fmriprep-23-2-0.sif \
	/data /out participant \
    --participant-label ${sub} \
	--output-space fsaverage5 MNI152NLin2009cAsym:res-2 \
	-w /out/workdir \
	--notrack \
    --fs-license-file /license/license.txt 