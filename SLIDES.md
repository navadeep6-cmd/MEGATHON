# Slide outline (10 slides)

Copy numbers from `reports/RESULTS.md` only.

1. **Title.** Automatic Category Suggestion for Wikimedia Commons. Team name, one-line pitch: "Explained, human-approved category suggestions for every new upload."
2. **Problem.** Millions of categories, a long tail, noisy and incomplete volunteer labels, and new images every second. Framed as extreme multi-label ranking for human review.
3. **Key insight.** We use all three approaches the organisers suggested. *Transfer learning*: a pre-trained CLIP model with a trained head. *Ensemble*: four scorers combined by a learned fusion model. *Multi-label*: each image gets several categories. Categories are treated as *retrievable objects*, not a fixed softmax, so new images and categories work immediately with no retraining.
4. **Architecture.** The diagram from the README: four scorers, candidate union, calibrated fusion, hierarchy post-processing.
5. **Data and leakage control.** Seeds across object, scene, event and abstract topics and many regions. Label cleaning. Temporal, uploader-disjoint split. dHash and embedding dedup. Use the table from `data_stats.json`.
6. **Main results.** The system comparison table: each scorer alone vs fusion vs fusion + hierarchy. Highlight P@1, R@10 and mAP.
7. **Long tail and breadth.** The band table (head, torso, tail, zero-shot) and the category-type table. Point out where retrieval and text prototypes help.
8. **Robustness and efficiency.** The corruption table, latency p50/p95, throughput, index size, and the 10⁸-image projection with PQ.
9. **Continual learning and transparency.** The demo screenshot with the "why" column and neighbour gallery. The feedback loop and drift monitor. The region and script audit.
10. **Limitations and next steps.** Heuristic type labels, the seed-based sample, missing-label effects (strict vs reviewed score), multilingual category names (Wikidata labels), and INT8 export.
