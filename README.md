# sea-ice-binary-ai4seaice
 Scripts for training and evaluating semantic segmentation CNNs for water-ice classification of Sentinel-1 images

## References
Please cite us!

The code in this repository was the main source for a sudy published as a [peer-reviewed paper](https://ieeexplore.ieee.org/abstract/document/10312772) (or the [preprint](https://eartharxiv.org/repository/view/6568/)). The bibtex entry for the paper is below:

```bibtex
@ARTICLE{sea_ice_unc_2023,
  author={Pires de Lima, Rafael and Karimzadeh, Morteza},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Model Ensemble With Dropout for Uncertainty Estimation in Sea Ice Segmentation Using Sentinel-1 SAR}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3331276}
}
```

The uncertainty analysis for sea ice classification was only part of a series of published studies on machine learning for sea ice. Code in [https://github.com/geohai/sea-ice-segment](https://github.com/geohai/sea-ice-segment) might also be of interest. Other than uncertainty analysis, we talked about:

* [Enhancing sea ice segmentation in Sentinel-1 images with atrous convolutions](https://www.tandfonline.com/doi/citedby/10.1080/01431161.2023.2248560?scroll=top&needAccess=true) ([preprint](https://arxiv.org/abs/2310.17122))
* [Comparison of Cross-Entropy, Dice, and Focal Loss for Sea Ice Type Segmentation](https://ieeexplore.ieee.org/abstract/document/10282060) ([preprint](https://arxiv.org/abs/2310.17135))
* [Deep Learning on SAR Imagery: Transfer Learning Versus Randomly Initialized Weights](https://ieeexplore.ieee.org/abstract/document/10281892) ([preprint](https://arxiv.org/abs/2310.17126))

The `bibtex` entry for the publications above is listed below:

```bibtex
@ARTICLE{enhancing_sea_ice_2023,
    author = {Rafael Pires de Lima, Behzad Vahedi, Nick Hughes, Andrew P. Barrett, Walter Meier and Morteza Karimzadeh},
    title = {Enhancing sea ice segmentation in Sentinel-1 images with atrous convolutions},
    journal = {International Journal of Remote Sensing},
    volume = {44},
    number = {17},
    pages = {5344-5374},
    year = {2023},
    publisher = {Taylor & Francis},
    doi = {10.1080/01431161.2023.2248560},
}
```

```bibtex
@INPROCEEDINGS{comparison_sea_ice_2023,
  author={de Lima, Rafael Pires and Vahedi, Behzad and Karimzadeh, Morteza},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Comparison of Cross-Entropy, Dice, and Focal Loss for Sea Ice Type Segmentation}, 
  year={2023},
  pages={145-148},
  address = {Pasadena, CA, USA}
  doi={10.1109/IGARSS52108.2023.10282060}
}
```

```bibtex
@INPROCEEDINGS{tl_sea_ice_2023,
  author={Karimzadeh, Morteza and de Lima, Rafael Pires},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Deep Learning on SAR Imagery: Transfer Learning Versus Randomly Initialized Weights}, 
  year={2023},
  pages={1983-1986},
  address = {Pasadena, CA, USA}
  doi={10.1109/IGARSS52108.2023.10281892}
}
```

