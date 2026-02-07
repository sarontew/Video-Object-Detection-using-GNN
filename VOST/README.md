VOST dataset: Video Object Segmentation under Transformations
============================================================================================

Package containing the training and validations sets of `VOST` dataset. Test set will be released separately.
More information on the [dataset website](https://www.vostdataset.org).

Documentation
-----------------

The directory is structured as follows:

 * `ROOT/JPEGImages`: Set of video sequences provided in the form of JPEG images sampled at 5 fps.

 * `ROOT/Annotations`: Set of manually annotated multiple-object ground-truth segmentations
   for objects undergoing transformations. Annotations are available at 5 fps.

 * `ROOT/Videos`: Set of raw videos in the MP4 format. Please use VLC to play the videos.

 * `ROOT/ImageSets`: Files containing the set of sequences on each of the dataset subsets.

 * `ROOT/JPEGImages_10fps`: Additional set of validation video sequences provided in the form of JPEG images sampled at 10 fps.

Credits
---------------

All sequences are sourced from [Ego4D](https://ego4d-data.org/) and [EPIC-KITCHENS](https://epic-kitchens.github.io/) and are licensed under Creative Commons Attributions 4.0 License, see [Terms of Use].

Citation
--------------

Please cite VOST, Ego4D and EPIC-KITCHENS in your publications if our data help your research:

    `@inproceedings{tokmakov2022breaking,
  title={Breaking the “Object” in Video Object Segmentation},
  author={Tokmakov, Pavel and Li, Jie and Gaidon, Adrien},
  booktitle={arXiv},
  year={2022}
  }`


    `@inproceedings{grauman2022ego4d,
  title={{Ego4D}: Around the world in 3,000 hours of egocentric video},
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and others},
  booktitle={CVPR},
  year={2022}
  }`

    `@article{damen2022rescaling,
  title={Rescaling egocentric vision: collection, pipeline and challenges for {EPCI-KITCHENS-100}},
  author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria and Furnari, Antonino and Kazakos, Evangelos and Ma, Jian and Moltisanti, Davide and Munro, Jonathan and Perrett, Toby and Price, Will and others},
  journal={International Journal of Computer Vision},
  volume={130},
  number={1},
  pages={33--55},
  year={2022},
  publisher={Springer}
  }`

Terms of Use
--------------

`VOST` is released under the Creative Commons License:
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Contact
--------------

support@vostdataset.org