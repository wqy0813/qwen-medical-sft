# Third-Party Notices

This file records third-party datasets and model assets used by the project. It does **not** replace the license terms supplied by each upstream owner. The project does not include raw dataset files, pretrained model weights, or trained model adapters in the GitHub source release.

## 1. MedMCQA

- Upstream dataset: [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- Paper: Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering*. Proceedings of the Conference on Health, Inference, and Learning, PMLR 174:248–260, 2022. [Paper and citation](https://proceedings.mlr.press/v174/pal22a.html).
- License metadata: the Hugging Face dataset repository declares **Apache-2.0** in its repository metadata.
- Important qualification: the dataset card's detailed “Licensing Information” section is incomplete, and its source-data description refers to historical exam questions collected from official websites and books. This project therefore records the upstream metadata declaration but does not make a broader legal determination about every underlying exam item. Users should review the upstream terms and any applicable source-material restrictions before redistribution or commercial use.
- This project does not redistribute the dataset content in this repository. Dataset paths, counts, and SHA256 values are recorded in `data_release_manifest.json`.

## 2. MedExpQA

- Upstream dataset: [HiTZ/MedExpQA](https://huggingface.co/datasets/HiTZ/MedExpQA)
- Code and data repository: [hitz-zentroa/MedExpQA](https://github.com/hitz-zentroa/MedExpQA)
- Paper: Iñigo Alonso, Maite Oronoz, and Rodrigo Agerri. *MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering*. Artificial Intelligence in Medicine, 2024, article 102938. [DOI: 10.1016/j.artmed.2024.102938](https://doi.org/10.1016/j.artmed.2024.102938).
- License declared by the Hugging Face dataset repository: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- Attribution: cite the paper above and identify the HiTZ/MedExpQA dataset as the source. Indicate project-side modifications where applicable.
- This project uses the English subset for evaluation only and does not redistribute the dataset content in this repository.

## 3. Qwen2.5-3B-Instruct

- Upstream model: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Pinned revision used by this project: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- License at the pinned revision: **Qwen Research License Agreement**, not Apache-2.0.
- The license grants use, reproduction, distribution, and modification for **non-commercial purposes only**. Commercial use requires a separate license from Alibaba Cloud.
- The license also sets conditions for redistribution of model materials or derivatives, including providing the agreement and retaining the required attribution notice. If a Qwen-derived model is distributed or made available, review the license's “Built with Qwen” / “Improved using Qwen” documentation requirement.
- [Pinned LICENSE file](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE).
- This source release does not include Qwen model weights or the project's trained adapter.

## 4. Qwen3.5-4B

- Upstream model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- Pinned revision used by this project: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- License declared by the upstream model repository at the pinned revision: **Apache License 2.0**.
- [Pinned model repository](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).
- This source release does not include Qwen model weights or the project's trained adapter.

## 5. Project-owned code and documentation

Original code and documentation authored for this project are licensed under the root-level [MIT License](../../LICENSE), except where a file states otherwise. The MIT License does not relicense third-party datasets, model weights, model outputs, or upstream documentation.

Before redistributing any model weights, LoRA adapters, dataset files, or other third-party materials, check the applicable upstream license and preserve all required notices. This file is a provenance summary, not legal advice.

## 6. Citation entries

### MedMCQA

```bibtex
@InProceedings{pmlr-v174-pal22a,
  title = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author = {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  pages = {248--260},
  year = {2022},
  volume = {174},
  series = {Proceedings of Machine Learning Research},
  url = {https://proceedings.mlr.press/v174/pal22a.html}
}
```

### MedExpQA

```bibtex
@article{ALONSO2024102938,
  title = {MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering},
  author = {Alonso, Iñigo and Oronoz, Maite and Agerri, Rodrigo},
  journal = {Artificial Intelligence in Medicine},
  pages = {102938},
  year = {2024},
  doi = {10.1016/j.artmed.2024.102938}
}
```

### Qwen2.5

```bibtex
@misc{qwen2.5,
  title = {Qwen2.5: A Party of Foundation Models},
  author = {{Qwen Team}},
  month = {September},
  year = {2024},
  url = {https://qwenlm.github.io/blog/qwen2.5/}
}
```

### Qwen3.5

```bibtex
@misc{qwen3.5,
  title = {Qwen3.5: Towards Native Multimodal Agents},
  author = {{Qwen Team}},
  month = {February},
  year = {2026},
  url = {https://qwen.ai/blog?id=qwen3.5}
}
```
