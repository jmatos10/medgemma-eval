# References

Works cited by *Does medical pretraining earn its keep? A controlled
evaluation of MedGemma 1.5 4B.*

Every entry was checked against the source on 2026-07-27. The record of what
was corrected during that check is in `DEVIATIONS.md` entry D-014.

---

## Data

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., Ni, B.
**MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical
image classification.** *Scientific Data*, 10:41, 2023.
doi:10.1038/s41597-022-01721-8

MedMNIST+. **18 standardized datasets for 2D and 3D biomedical image
classification with multiple size options: 28, 64, 128, and 224.** Zenodo,
doi:10.5281/zenodo.10519652

> The 2023 paper describes the 28x28 release. The 224x224 images used here
> come from the MedMNIST+ extension. Both are cited when the data is
> described.

Tschandl, P., Rosendahl, C., Kittler, H. **The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions.** *Scientific Data*, 5:180161, 2018. Source dataset for DermaMNIST.

Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., Rodellar, J.
**A dataset of microscopic peripheral blood cell images for development of
automatic recognition systems.** *Data in Brief*, 30:105474, 2020. Source
dataset for BloodMNIST.

---

## Models

Sellergren, A., Gao, C., Mahvar, F., Kohlberger, T., Jamil, F., Traverse,
M., Tono, A., Sadjad, B., Yang, L., Lau, C., et al. **MedGemma 1.5 Technical
Report.** arXiv:2604.05081, 2026.

> The model evaluated in arms A1 through A4. This is the citation the model
> card for `google/medgemma-1.5-4b-it` specifies. Source of the pretraining
> modality list on which H3 rests, and of the zero-shot anchors DermMCQA
> 73.5 and PathMCQA 70.0.

```bibtex
@article{sellergren2026medgemma,
  title={MedGemma 1.5 Technical Report},
  author={Sellergren, Andrew and Gao, Chufan and Mahvar, Fereshteh and
  Kohlberger, Timo and Jamil, Fayaz and Traverse, Madeleine and Tono,
  Alberto and Sadjad, Bashir and Yang, Lin and Lau, Charles and others},
  journal={arXiv preprint arXiv:2604.05081},
  year={2026}
}
```

Sellergren, A., et al. **MedGemma Technical Report.** arXiv:2507.05201,
2025. The earlier MedGemma 4B and 27B release. Cited for lineage only; it
does not describe the checkpoint evaluated here.

Gemma Team, Google DeepMind. **Gemma 3 Technical Report.** arXiv:2503.19786,
2025. The base model, and the control arm A5.

He, K., Zhang, X., Ren, S., Sun, J. **Deep residual learning for image
recognition.** *CVPR*, 2016. ResNet-18, arm A6.

---

## Methods

Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
Chen, W. **LoRA: Low-rank adaptation of large language models.** *ICLR*,
2022. arXiv:2106.09685. The adaptation method used in arms A4 and A5.

Holtzman, A., West, P., Shwartz, V., Choi, Y., Zettlemoyer, L. **Surface
form competition: Why the highest probability answer isn't always right.**
*EMNLP*, 2021. arXiv:2104.08315, ACL Anthology 2021.emnlp-main.564. Basis
for the PMI scoring variant reported as a robustness check in arm A3.

Wu, Y., et al. **Google's neural machine translation system: Bridging the
gap between human and machine translation.** arXiv:1609.08144, 2016. Source
of length normalization in sequence scoring, the basis for the prespecified
mean log-probability scoring in arm A3.

Efron, B. **Bootstrap methods: Another look at the jackknife.** *Annals of
Statistics*, 7(1):1-26, 1979. All confidence intervals reported here are
1,000-resample percentile bootstraps.

---

## Related evaluations

Buskila, A.-a. A. **Domain Fine-Tuning vs. Retrieval-Augmented Generation
for Medical Multiple-Choice Question Answering: A Controlled Comparison at
the 4B-Parameter Scale.** arXiv:2604.23801, Bar-Ilan University, April 2026.

> Direct prior work. Compares Gemma 3 4B against MedGemma 4B, 4-bit
> quantized, on the MedQA-USMLE 4-option test split (1,273 questions, three
> repetitions), finding +6.8 percentage points in majority-vote accuracy
> from domain fine-tuning and no significant gain from RAG. This is Q2 in
> text-only form. Three differences here: a vision task rather than text
> multiple choice, LoRA at a matched budget rather than quantized
> off-the-shelf inference, and a modality-inside versus modality-outside
> contrast, which that study does not run.

Gosai, A., Kavishwar, A., McNamara, S. L., Samineni, S., Umeton, R.,
Chowdhury, A., Lotter, W. **Beyond Diagnosis: Evaluating multimodal LLMs for
pathology localization in chest radiographs.** arXiv:2509.18015, 2025.

> Compares MedGemma 27B against Gemma 3 27B to separate model size from
> domain-specific training. Reports MedGemma at 17.7 percent localization
> accuracy against GPT-5 at 49.7, GPT-4 at 39.1, a CNN baseline at 59.9, and
> radiologists at 80.1. The CNN exceeding both language models is consistent
> with H4.

Chung, H.-H., Li, S., Wald, Y., Han, X., Saria, S., Ghosh, J. **MILM: Large
Language Models for Multimodal Irregular Time Series with Informative
Sampling.** arXiv:2605.13711v1 [cs.LG], 13 May 2026.

> Appendix F, "MedGemma vs. Qwen," reports zero-shot unparseable output
> rates for the 4B instruction-tuned variants:
>
> | Dataset | MedGemma-4B | Qwen3-4B |
> |---|---|---|
> | MIMIC-IV-IHM | 62.70% | 0.00% |
> | MIMIC-IV-LOS | 89.66% | 0.00% |
> | eICU-IHM | 65.70% | 0.00% |
> | eICU-LOS | 80.64% | 0.00% |
>
> The named failure mode is emitting `A` instead of `<answer> A </answer>`,
> violating the format specified in the prompt. Unparseable responses are
> assigned a neutral score of 0.5, and MedGemma-4B underperforms Qwen3-4B on
> all eight metric-dataset combinations.
>
> Independent evidence for the phenomenon behind arm A1. Three caveats when
> citing it: the task is text-only clinical time series rather than images,
> the checkpoint is MedGemma-4B (their reference [79] is arXiv:2507.05201,
> the v1 report) rather than MedGemma 1.5, and their response was to switch
> models while the response here is to change the measurement.
>
> **Their stated explanation is directly testable, and this study tests it.**
> They suspect MedGemma underperformed because its "domain-specific training
> is primarily oriented toward medical image understanding, with text
> supervision derived from relatively small medical QA datasets." If that
> were the whole story, the parsing failure should be milder on image tasks.
> It is not: this study measures 100 percent unparseable across 2,715
> zero-shot attempts on two image classification tasks, the task type their
> explanation predicts MedGemma should handle well.
>
> Their MILM fine-tuning used QLoRA at rank 16, alpha 16, dropout 0.05 on
> query, key, value, and output projections. The configuration here is rank
> 16, alpha 32, dropout 0.05 on attention plus MLP projections. Arrived at
> independently from standard defaults; noted for comparison.


