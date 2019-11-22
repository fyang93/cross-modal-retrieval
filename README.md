PyTorch code for Mining on Heterogeneous Manifolds for Zero-shot Cross-modal Image Retrieval

Reference:
- [F. Yang](https://fyang.me), [Zheng Wang](https://wangzwhu.github.io/home), Jing Xiao, [Shin'ichi Satoh](http://research.nii.ac.jp/~satoh/index.html)

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
