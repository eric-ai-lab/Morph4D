
<div align="center">

<h3>MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator</h3>

👤 Xuehai He* · Shijie Zhou* · Thivyanth Venkateswaran · Kaizhi Zheng · Ziyu Wan · Achuta Kadambi · Xin Eric Wang

🔗 <a href="https://arxiv.org/abs/2510.04390">arXiv</a> &nbsp;|&nbsp; 🎬 <a href="#MorphoSim.mp4">Demo Video</a>

</div>


## TODO
- [ ] Gradio Demo
- [ ] Huggingface setup
- [ ] Release training code
- [ ] Release inference code
- [ ] Release demo


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

## Demo video

<video controls width="720" muted playsinline poster="morphosim_thumbnail.jpg">
  <source src="MorphoSim.mp4" type="video/mp4">
  <a href="MorphoSim.mp4">direct link</a>.
</video>


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
