Citing and acknowledgments
==========================

Citing HABIT
------------

If you use HABIT in scientific work, please cite the software with the
version you ran. The citation metadata is in ``CITATION.cff`` at the
repository root (GitHub shows it as "Cite this repository"):

   Li C, Dong M, HABIT Contributors. *HABIT: Habitat Analysis - Biomedical
   Imaging Toolkit*. Version |release|. Apache-2.0.
   https://github.com/lichao312214129/HABIT

No DOI has been registered for the software yet; cite the repository URL
and the exact version until one exists.

For the methods section, let HABIT draft the paragraph from what actually
ran rather than from memory:

.. code-block:: python

   result = Study(spec).fit_predict(cohort)
   print(result.manifest.describe_methods())

It states the executed stages with their parameters, the seed, software
versions and any excluded subjects. A worked example is
:doc:`/auto_examples/05_validation_and_reuse/plot_01_train_save_predict`.
Review and edit it like any draft.

Acknowledgments
---------------

We sincerely thank the following experts and institutions for their support and contributions to the HABIT project.

Core developers
~~~~~~~~~~~~~~~

**Li Chao**

* Institution: Yongchuan Hospital, Chongqing Medical University

**Dong Mengshi**

* Institution: The Third Affiliated Hospital of Sun Yat-sen University

**Jiang Zekun**

* Institution: West China Biomedical Big Data Center, West China Hospital, Sichuan University

**Jiang Kunyuan**

* Institution: West China Medical Robotics Center

Future developers
~~~~~~~~~~~~~~~~~

The following members are expected to join HABIT development.

Clinical collaborators
~~~~~~~~~~~~~~~~~~~~~~

**Dr. Huang Teng**

* Institution: Ganzhou People's Hospital

**Dr. Wang Yaqi**

* Institution: Anhui Medical University Third Affiliated Hospital

**Dr. Li Mengsi**

* Institution: Central South University Xiangya Hospital

**Dr. Tang Junjie**

* Institution: Southern Medical University Zhujiang Hospital
* Special contributions: Image registration algorithm design, implementation, and validation (including collaborative test code).

Special thanks
~~~~~~~~~~~~~~

We thank all experts, scholars, and users who have provided valuable feedback in clinical applications, algorithm improvements, and project optimization. Your support drives HABIT's continued development.

Open source community
~~~~~~~~~~~~~~~~~~~~~

HABIT relies on many excellent open-source projects, including but not limited to:

**SimpleITK** — Medical image processing
**PyRadiomics** — Radiomics feature extraction
**scikit-learn** — Machine learning algorithms
**pandas** & **numpy** — Data processing
**matplotlib** & **seaborn** — Visualization

We extend our sincere respect and gratitude to the developers and maintainers of these projects.

How to contribute
~~~~~~~~~~~~~~~~~

To contribute code, docs, or extensions, start with
:doc:`development/architecture`, then :doc:`development/contributing`
(environment, tests, and pull-request conventions), then
:doc:`customization/index`. Bug reports and feature requests belong on GitHub Issues:
|link_github_issues|.

Contact
~~~~~~~

For collaboration or other inquiries:

**Email**: lichao19870617@163.com
**GitHub Issues**: |link_github_issues|

.. note::
   If you have used HABIT in your research and would like to be acknowledged, please contact us via GitHub Issue or email.
