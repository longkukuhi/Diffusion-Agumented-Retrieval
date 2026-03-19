# Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval

This repository is the official implementation of [Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval](https://dl.acm.org/doi/abs/10.1145/3726302.3729950), published as a full paper at SIGIR 2025.

## Dataset
For evaluation, please prepare the following datasets.
- [COCO 2017 Unlabeled images](https://cocodataset.org/#download)
- [VisDial](https://visualdialog.org/data)

If you want to evaluate DAR with generated-image augmentation, please download the generated images:

- [val_generated_images](https://drive.google.com/file/d/1ifR5q1ttOGxz-CXJ0ZR7cTsrh8FHLlbP/view)


## Checkpoints
Please download the weight of the [BLIP model](https://github.com/salesforce/BLIP) and the [ChatIR Checkpoints](https://github.com/levymsn/ChatIR?tab=readme-ov-file). 

## Suggested Project Structure
```text
your_project/
├── eval.py
├── baselines.py
├── chatir_weights.ckpt
├── BLIP/
│   ├── configs/
│   ├── models/
│   └── ...
├── dialogues/
│   └── VisDial_v1_0_queries_val.json
├── ChatIR_Protocol/
│   └── Search_Space_val_50k.json
├── val_generated_images/
│   ├── 0_0.jpg
│   ├── 0_1.jpg
│   ├── ...
│   └── N_10.jpg
└── temp/
```

## Evaluation

All experiments can be run with `eval.py`.

```bash
python eval.py
```

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

## Acknowledgement

We thank the authors of BLIP and ChatIR for making their code publicly available.
