# Model card: Commons category suggester

**Intended use.** The system suggests Wikimedia Commons categories for newly uploaded files, for review by human editors. It is not meant for automatic editing, and not for identifying people.

**Model.**
- Encoder: OpenCLIP ViT-B/16 (`laion2b_s34b_b88k`), open weights, frozen.
- Supervised head: linear, one output per category with ≥ 10 training images. Trained with asymmetric loss (γ⁺ = 0, γ⁻ = 4, margin 0.05) and initialised from the category-name text embeddings.
- Retrieval: cosine k-NN (K = 50, τ = 0.05) over categorised images.
- Metadata: text-tower embedding of the title and description, lexical overlap, year match, and a geo vote over a 30 km radius.
- Fusion: logistic regression on standardised features, fitted on the validation split.

**Data.**
- Commons files crawled through the public MediaWiki API from seed categories covering objects, scenes, events, abstract concepts and 15 cities across Africa, Asia and Latin America.
- Only openly licensed files are used.
- Removed from the labels: maintenance, tracking, provenance, licence and assessment categories, soft-redirected categories, and categories of individual people.
- Split: by upload time, uploader-disjoint, with near-duplicates removed.

**Evaluation.**
- Metrics: sample-wise P@k, R@k, F1@k and mAP; macro (per-category) metrics; hierarchy-aware precision.
- Stratified by frequency band, category type, label script, region and resolution.
- Robustness under blur, JPEG, low resolution and cropping.
- Results are in `reports/RESULTS.md`.

**Known biases and risks.**
- Commons is dominated by European and North American uploads, and the encoder's training data (LAION) has similar skews. The region and script audits make this visible; they do not remove it.
- Missing labels make strict scores pessimistic. The review procedure (`s12_apply_review.py`) estimates by how much.
- Category type and region are heuristic labels, used only for reporting.

**Safety.** Categories of individual people are excluded, every suggestion requires human approval, and nothing is written back to Commons.
