<div align="center">

<h2>RoPEMover: Depth-Aware Object Relocation via Positional Embeddings</h2>

[İpek Öztaş](https://ipekoztas.github.io/)<sup>1,3</sup> |
[Duygu Ceylan](https://www.duygu-ceylan.com/)<sup>2</sup> |
[Aybars Buğra Aksoy](https://www.linkedin.com/in/aybars-bugra-aksoy/)<sup>1</sup> |
[Ayşegül Dündar](https://www.cs.bilkent.edu.tr/~adundar/)<sup>1</sup>

<sup>1</sup>Bilkent University, <sup>2</sup>Adobe Research, <sup>3</sup>Brown University

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ipekoztas.github.io/RoPEMover/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.27332)&nbsp;
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-demo-blue)](#)

</div>

---

## 📰 Abstract

Moving an object in a single image requires geometry-consistent spatial
rearrangement, including handling occlusions, revealing previously unseen
regions, and maintaining coherent shadows and reflections. Existing
approaches are not well suited to this setting and often fail to preserve
such scene-level consistency.

We address this problem by introducing a geometry-aware object motion
method that operates directly on the positional representations of
diffusion transformers. Our key insight is that rotary positional
embeddings (RoPE) define a structured spatial field that can be
explicitly manipulated to induce controlled motion. We extend 2D RoPE
into a depth-aware formulation that encodes 3D spatial structure,
enabling consistent object displacement and scene-aware updates.

Our model is trained using synthetic data combined with a small set of
real images via parameter-efficient fine-tuning. Despite minimal real
supervision, it preserves object identity under large spatial
displacements, generates plausible content in newly revealed regions,
and consistently updates scene-dependent effects such as shadows and
illumination.

Experimental results on standard object motion benchmarks demonstrate
state-of-the-art performance across all evaluation metrics.

---

## ⏰ Updates

- [x] **2026.6.25**: Project page released at
  [`ipekoztas.github.io/RoPEMover`](https://ipekoztas.github.io/RoPEMover/).
- [ ] Release arXiv preprint.
- [ ] Release inference code and pretrained weights.
- [ ] Release interactive Hugging Face Space demo.

---

## 🔗 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{oztas2026ropemover,
  title   = {RoPEMover: Depth-Aware Object Relocation via Positional Embeddings},
  author  = {Oztas, Ipek and Ceylan, Duygu and Aksoy, Aybars Bugra and Dundar, Aysegul},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## ©️ License

This project will be released under the
[Apache 2.0 license](LICENSE) once the codebase is published.

## 📧 Contact

For questions, please reach out to
[ipek.oztas@bilkent.edu.tr](mailto:ipek.oztas@bilkent.edu.tr).
