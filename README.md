# Diffusion Augmented Retrieval (DAR)

Official implementation of *Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval*, published at SIGIR 2025. Given successive turns of a dialogue, DAR combines a text-to-image retrieval score with a similarity score from a diffusion-generated query image. [Read the paper](https://doi.org/10.1145/3726302.3729950).

## Prepare the evaluation

| Input | Where it goes |
| --- | --- |
| [COCO 2017 unlabeled images](https://cocodataset.org/#download) | The candidate images must resolve at the paths recorded in [the 50k candidate list](ChatIR_Protocol/Search_Space_val_50k.json). |
| [VisDial](https://visualdialog.org/data) queries | The validation [dialogue JSON](dialogues/VisDial_v1_0_queries_val.json) is included. Each target image path must match an entry in the candidate list. |
| [ChatIR retrieval weights](https://github.com/levymsn/ChatIR) | Place the checkpoint at `chatir_weights.ckpt` in the repository root. [baselines.py](baselines.py) loads it into the bundled BLIP model. |
| [Generated validation images](https://drive.google.com/file/d/16enUJKzibsby05rcSBxyiC6jpPUruWf-/view?usp=sharing) | Extract to `val_generated_images/`. Each dialogue needs 11 files named `<dialogue_index>_<round>.jpg` for rounds 0–10. |

## Compare text-only and augmented retrieval

From the repository root, install PyTorch and its matching torchvision build for your device, then install the [BLIP dependencies](BLIP/requirements.txt) and run:

```bash
python -m pip install -r BLIP/requirements.txt
python -m pip install tqdm Pillow
python eval.py
```

[eval.py](eval.py) runs the text-only baseline first and the augmented method second. The complete command needs the generated validation images for the second pass. It prints cumulative Hits@10 across dialogue rounds. The augmented pass assigns the generated-image score a weight of 0.2 for rounds 0 and 1 and 0.5 thereafter.

The evaluator saves corpus and generated-image embeddings under `temp/`. Use a fresh cache directory or separate cache filenames in `build_model` when changing the checkpoint, candidate list or generated images. The filenames do not encode those inputs.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{long2025diffusion,
  title={Diffusion augmented retrieval: A training-free approach to interactive text-to-image retrieval},
  author={Long, Zijun and Liang, Kangheng and Aragon Camarasa, Gerardo and Mccreadie, Richard and Henderson, Paul},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={823--832},
  year={2025}
}
```

## Acknowledgments

We thank the authors of [BLIP](https://github.com/salesforce/BLIP) and [ChatIR](https://github.com/levymsn/ChatIR) for releasing their retrieval models and code.
