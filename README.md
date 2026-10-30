# Video-Based Sexual Harassment Detection Using Fine-Tuned Video Transformers

**Authors:** Jubayer Al Mahmud\*, Md Shamimul Islam\*, Md Nur Hossain

\* Equal contribution (co-first authors)

Official code for the paper presented at the **IEEE 3rd International Conference on Computing, Applications and Systems (COMPAS 2026)**, 9-10 October 2026, University of Dhaka, Bangladesh. [arXiv:2610.15354](https://arxiv.org/abs/2610.15354)

Many sexual harassment incidents go unreported because evidence is limited and help arrives late. Automatic detection in surveillance video could support faster intervention, but it is a hard vision problem: clothing, lighting, occlusion, low resolution and camera angle vary widely, and there are very few annotated video datasets. In this work we extend the SHV (Sexual Harassment Video) dataset to 600 clips and fine-tune three video transformers pretrained on Kinetics-400 to classify each clip as **Harassment** or **Non_Harassment**.

## Results

Test set of 181 videos (extended SHV dataset).

| Model | Pretrained checkpoint | Frames per clip | Test Accuracy |
|---|---|---|---|
| **TimeSformer** | `facebook/timesformer-base-finetuned-k400` | 16 | **92.27%** |
| VideoMAE | `MCG-NJU/videomae-base-finetuned-kinetics` | 16 | 91.71% |
| ViViT | `google/vivit-b-16x2-kinetics400` | 32 | 91.16% |

All three models reach similar accuracy. In our experiments, performance depended on dataset size, video quality, the frame sampling strategy and the model architecture. Confusion matrices, ROC curves and training curves are saved in the notebook outputs.

## Repository Structure

```
├── notebooks/
│   ├── TimeSformer.ipynb   # fine-tuning and evaluation of TimeSformer
│   ├── VideoMAE.ipynb      # fine-tuning and evaluation of VideoMAE
│   └── ViViT.ipynb         # fine-tuning and evaluation of ViViT
├── requirements.txt
└── LICENSE
```

Each notebook follows the same steps:
1. Install libraries and load the pretrained model
2. Load and check the dataset (unreadable videos are skipped)
3. Build the dataset loader with uniform frame sampling
4. Replace the classification head with a 2-class output layer
5. Train for 20 epochs (AdamW, lr 1e-5, weight decay 1e-4, batch size 2), saving checkpoints after every epoch
6. Evaluate the best checkpoint on the test set (classification report, confusion matrix, ROC curve) and run a prediction on a single video

## Dataset

The extended SHV dataset has 600 short YouTube clips, labeled Harassment or Non_Harassment, split into 419 training and 181 test videos. The videos are **not** included in this repository.

The notebooks expect a zip file with this layout:

```
Sexual_Harassment_Dataset/
├── train/
│   ├── Harassment/
│   └── Non_Harassment/
└── test/
    ├── Harassment/
    └── Non_Harassment/
```

## How to Run

The notebooks were run on **Google Colab** with a GPU runtime.

1. Upload `Sexual_Harassment_Dataset.zip` to your Google Drive.
2. Open a notebook in Colab and set **Runtime > Change runtime type > GPU**.
3. Update the Drive paths in the notebook (`/content/drive/MyDrive/ModelTrain/Sexual_Harassment/...`) to where you keep the dataset and want checkpoints saved.
4. Run all cells. Training can resume from `last_checkpoint.pth` if the Colab session disconnects.

To run locally instead:

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@inproceedings{mahmud2026harassment,
  title     = {Video-Based Sexual Harassment Detection Using Fine-Tuned Video Transformers},
  author    = {Al Mahmud, Jubayer and Islam, Md Shamimul and Hossain, Md Nur},
  booktitle = {IEEE 3rd International Conference on Computing, Applications and Systems (COMPAS)},
  address   = {Dhaka, Bangladesh},
  year      = {2026},
  note      = {arXiv:2610.15354. Jubayer Al Mahmud and Md Shamimul Islam contributed equally}
}
```

## License

The code is released under the MIT License, see [LICENSE](LICENSE). The pretrained models are subject to their own licenses on Hugging Face.
