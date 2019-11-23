PyTorch code for Mining on Heterogeneous Manifolds for Zero-shot Cross-modal Image Retrieval

Reference:
- [F. Yang](https://fyang.me), [Zheng Wang](https://wangzwhu.github.io/home), Jing Xiao, [Shin'ichi Satoh](http://research.nii.ac.jp/~satoh/index.html), "Mining on Heterogeneous Manifolds for Zero-shot Cross-modal Image Retrieval", AAAI 2020.

## Prepare

Download the SYSU-MM01 dataset and run `make preprocess` to prepared the files needed for training.

## Training

We train two single-modal models as well as a cross-modal model.
Train single-model by running `make train`, you may want to edit argument `modal` to train model for specific modality.
Train cross-modal by running `make train_cross`.
Set `CUDA_VISIBLE_DEVICES` according to your needs.

## Testing

Use `make test` to evaluate.
Set `CUDA_VISIBLE_DEVICES` according to your needs.

## Models

You may find our trained models below.
- [Model for thermal images](https://drive.google.com/file/d/160HbLLCq5-sm78ItA4pS347MJc5s0_tS/view?usp=sharing)
- [Model for visible images](https://drive.google.com/file/d/1tT0uCpn0aY5kUS2hrmKcoLr5VYg4eZlu/view?usp=sharing)
- [Cross-modal model for both modalities](https://drive.google.com/file/d/1HTUctPdIJDpo4cJOThk9eUeMEcybUlhO/view?usp=sharing)
