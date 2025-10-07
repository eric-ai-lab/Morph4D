# MorphoSim

MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator


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
```bibtex
@misc{he2025morphosiminteractivecontrollableeditable,
      title={MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator}, 
      author={Xuehai He and Shijie Zhou and Thivyanth Venkateswaran and Kaizhi Zheng and Ziyu Wan and Achuta Kadambi and Xin Eric Wang},
      year={2025},
      eprint={2510.04390},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.04390}, 
}
```
