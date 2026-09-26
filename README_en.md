# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**Language / 语言**：[English](https://github.com/lichao312214129/HABIT/blob/main/README_en.md) | [简体中文](https://github.com/lichao312214129/HABIT/blob/main/README.md)

## The standardization bottleneck in habitat imaging, and the open-source tool HABIT

Radiomics extracts a large number of quantitative features at high throughput from medical images (such as CT, MRI, and PET-CT) and relates those features to clinical outcomes, genotypes, or pathology. Since it was proposed in 2012, tens of thousands of papers have been published. Cell density, perfusion, necrosis, hypoxia, immune infiltration, and stromal response inside a tumor are not spatially uniform. The limitation of radiomics is that it treats the tumor as a whole and compresses a highly heterogeneous tumor into one set of global features, losing spatial information. In recent years, habitat imaging has been proposed. Its change in approach is to partition the tumor, at the voxel level, into subregions called habitats. Each habitat has similar physiological and pathological features. The method can quantify spatial heterogeneity of a tumor noninvasively, in vivo, before surgery. The idea comes from ecology. The core assumption is that habitats with similar image features share similar tumor biology. Different habitats correspond to different microenvironmental selection pressures and adaptive strategies. Some harbor more aggressive, harder-to-treat populations, and interactions between habitats can also change tumor biology. This method is called Radiomics++, an advanced or complementary form of radiomics. Hundreds of papers have been published and the number has grown exponentially in recent years, yet the field is still early. The workflow is complicated and lacks a standardized protocol and analysis tools. That has become the bottleneck preventing habitat imaging from becoming reproducible, verifiable, and ready for clinical translation. For this reason, we spent more than two years developing HABIT (Habitat Analysis: Biomedical Imaging Toolkit), an open-source habitat-imaging tool, to provide a standardized, reproducible, extensible, and embeddable analysis framework for habitat imaging.

## What HABIT is, and what it does

From our search of public repositories and the literature, there is still no complete open-source tool devoted to habitat imaging. The scattered code that exists is scripts for single studies, bound to that study's features and partitioning rules. It does not generalize, and it lacks documentation that can be used on its own. Closed commercial platforms in China have had substantial success in radiomics education and have been used in hundreds of papers. They support part of habitat analysis, but the core details are not open, cannot be inspected, cannot be extended, and are not easy to embed as a citable module in an existing radiomics ecosystem.

From the start, HABIT was constrained to be inspectable, reproducible, extensible, and embeddable in the ecosystem. The source and the computational settings can be checked. The same protocol can be reproduced on different cohorts. A step can be replaced. It can be used as a module inside an existing radiomics pipeline. The software is fully open source and comes with detailed documentation. It is about to enter public testing and presentation. After more than two years of development, it is about to enter full public testing and outreach. Source: https://github.com/lichao312214129/HABIT . Docs: https://lichao312214129.github.io/HABIT/ . Install with `pip install habitat-analysis`, or install from source.

```bash
pip install habitat-analysis
```

HABIT's functions fall into seven items.

1. Voxel-texture extraction aligned with academic standards, with very high time efficiency. Voxel-texture extraction in study-specific scripts and commercial platforms does not state alignment with IBSI, and it is slow. HABIT's feature formulas strictly follow the IBSI standard and are aligned with PyRadiomics. HABIT includes a CUDA-accelerated voxel-texture method. In a same-machine comparison (NVIDIA GeForce RTX 3070 Laptop, binWidth=25, kernelRadius=1, 80,084 voxels), the PyRadiomics voxel path took 414.58 seconds, and HABIT with CUDA took 7.59 seconds.

2. Voxel-feature preprocessing and habitat partitioning. Study-specific scripts and commercial platforms generally do not support preprocessing and selection of voxel features. Habitat partitioning usually offers only one or two rules. HABIT includes the three partitionings commonly used in the literature: supervoxel then cluster (two_step_habitat), within-case voxel clustering (one_step_habitat), and cohort-pooled voxel clustering (direct_pooling_habitat). Each step is provided as an atomic operation, and the user can compose strategies. fit on the training set yields shared preprocessing parameters, centroids, and assignment rules; predict on the test set generates habitat maps in the same feature space.

3. Voxel-feature stability analysis. Published studies show that a substantial fraction of voxel-texture features is unstable under retest or perturbation. HABIT includes a stable-feature selection module.

4. Habitat-map quantification. HABIT supports comprehensive quantitative feature extraction on habitat maps, including a full set of graph-theoretic features developed in this project. HABIT defines each feature strictly from published academic standards, and publishes all formulas and APIs in the documentation.

5. Extension interfaces. Every core module in HABIT can be replaced and extended, and the software as a whole can be embedded in an existing radiomics pipeline. Changing a step in a study-specific script requires rewriting that study's pipeline code. Commercial platforms wrap the pipeline in an interface or a job system, which is hard to embed in an existing preprocessing or statistics workflow. HABIT specifies replaceable steps through protocols and a registry. The upper interface is Study, HabitatSpec, and YAML. The lower interface accepts SimpleITK images, NumPy arrays, file paths, and lists of paths.

6. Cohort execution. HABIT supports continuous runs on large cohorts. Voxel extraction and per-case computation can use a process pool. After an interruption, the run continues from completed cases. A failed case is isolated and recorded, and does not stop the cohort.

7. Documentation. HABIT ships with online documentation covering the whole workflow. The HABIT docs cover installation, Quickstart, the Habitat Analysis Guide, and feature definitions (voxel texture, MSI, ITH, graph theory), with runnable examples.

## Citation and license

- Issues: [GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- Citation: [CITATION.cff](CITATION.cff) · [Acknowledgments](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- License: [Apache-2.0](LICENSE) for HABIT, except the consensus-clustering module at `habit/third_party/inmoose/`, which is GPL-3.0-or-later (InMoose / Sajovic). Keep the copyright and license notices and ship [NOTICE](NOTICE) with redistributions. Please cite HABIT when it supports scientific work.
