# Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval

This repository is the official implementation of[Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval](https://dl.acm.org/doi/abs/10.1145/3726302.3729950), published as a full paper at SIGIR 2025.

## Dataset
For evaluation, please prepare the following datasets.
- [COCO 2017 Unlabeled images](https://cocodataset.org/#download)
- [VisDial](https://visualdialog.org/data)

Suggested structure:
```text
data/
├── VisDial/
│   ├── train/
│   │   ├── images/
│   │   └── visdial_1.0_train.json
│   └── val/
│       ├── images/
│       └── visdial_1.0_val.json
```

## Checkpoints
Please download the weight of the [BLIP model](https://github.com/salesforce/BLIP) and the [ChatIR Checkpoints](https://github.com/levymsn/ChatIR?tab=readme-ov-file). 
Please place the required checkpoints under `ckpt/`.
Example:
```text
ckpt/
├── model_base_retrieval_coco.pth
├── chatir_weights.ckpt
└── ...
```

## Evaluation

All experiments can be run with `eval.py`.

```bash
python eval.py
```

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{long2025dar,
  author    = {Zijun Long and Kangheng Liang and Gerardo Aragon Camarasa and Richard Mccreadie and Paul Henderson},
  title     = {Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2025},
  doi       = {10.1145/3726302.3729950}
}
```

## Acknowledgement

We thank the authors of BLIP, ChatIR, and CLIP for making their code publicly available.
