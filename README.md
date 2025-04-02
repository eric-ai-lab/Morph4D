# Morph4D

Morpho4D: An Interactive, Controllable, and Editable Text-to-4D World Simulator


## TODO
- [ ] Gradio Demo
- [ ] Huggingface setup
- [ ] Release training code
- [ ] Release inference code


## Environment Setup
```
bash env.sh
```

## Checkpoint preparation
### LSeg model preparation
Download the LSeg model file `demo_e200.ckpt` using gdown and place it under the folder: `lseg_encoder`.
```
gdown 1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb
mv demo_e200.ckpt lseg_encoder
```

## E2E training + inference

Generate an editable 4D scene with prompt "A teal robot is cooking food in a kitchen with steam rising from pots"
```
bash run.sh e2e
```



## Citation
