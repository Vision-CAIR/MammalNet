# Mammal and their behavior recognition tasks

## Dataset prepare
onging


## Installation

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md). 

## Train the model with MViT

```
python tools/run_net_composition_augmentation.py --cfg configs/mammalnet/MVITv2_B_16x16_composition.yaml
```
## Train the model with SlowFast

```
python tools/run_net_composition.py --cfg configs/mammalnet/slowfast_32x8_composition.yaml
```


## Train the model with I3D

```
python tools/run_net_composition.py --cfg configs/mammalnet/i3d_16x24_composition.yaml
```

## Train the model with C3D

```
python tools/run_net_composition.py --cfg configs/mammalnet/C3D_16x24_composition.yaml
```


## Under few-shot set-up
take MViT as an example:

## Train the model with MViT

```
python tools/run_net_composition_augmentation.py --cfg configs/mammalnet/MVITv2_B_16x16_composition_zeroshot.yaml
python tools/run_net_composition_augmentation.py --cfg configs/mammalnet/MVITv2_B_16x16_composition_oneshot.yaml
python tools/run_net_composition_augmentation.py --cfg configs/mammalnet/MVITv2_B_16x16_composition_fiveshot.yaml
```