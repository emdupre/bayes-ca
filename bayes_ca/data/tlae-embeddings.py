import json
from pathlib import Path

import click
import whisper
import numpy as np
from sentence_transformers import SentenceTransformer


@click.command()
@click.option("--file")
@click.option("--datadir")
@click.option("--outdir")
def main(file, datadir, outdir):
    """
    This script assumes you have access to the copyrighted stimuli
    and that you are running in an environment with SBert (i.e.,
    sentence-transformers) and OpenAI's Whisper installed.

    Params
    ------
    file : str
        Stimulus file to transcribe and generate embedding
    datadir : str
        Local path to the stimuli files
    outdir : str
        Local path to store transcriptions and embeddings
    """
    model = whisper.load_model("medium.en")
    result = model.transcribe(str(Path(datadir, file)))

    whisper_outname = "whisper-" + str(Path(file).with_suffix(".json"))
    with open(Path(outdir, whisper_outname), "w") as outfile:
        json.dump(result, outfile, indent=4)

    sentences = result["text"].split(". ")
    model = SentenceTransformer("all-mpnet-base-v2")

    embeddings = model.encode(sentences)
    sbert_outname = Path(outdir, f"sbert-{whisper_outname}")
    np.save(sbert_outname, embeddings, allow_pickle=False)

    return


if __name__ == "__main__":
    main()
