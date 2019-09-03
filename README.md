# Applying the 'Style Transfer' algorithm to create a "Thai Modern Artwork"

This project contains my experiment  to apply the **Style Transfer** algorithm in Deep Learning to produce a new "Thai Modern Artwork".

See more details about this work in my medium post [(Style Transfer) How to create a "Thai Modern Artwork" without artistic skills](https://medium.com/@nongnoochr/style-transfer-how-to-create-a-thai-modern-artwork-without-artistic-skills-4462aa6eaa3)

### Table of Contents

* [Installation](#installation)
* [Project Details](#details)
* [Licensing, Authors, and Acknowledgements](#licensing)


## Installation<a name="installation"></a>
Below are python libraries that are required to run this code using Python versions 3.7.0:

* matplotlib
* numpy
* Pillow
* pytorch 1.0 (Stable) with CUDA 10.0 (See [here](https://thedavidnguyenblog.xyz/installing-pytorch-1-0-stable-with-cuda-10-0-on-windows-10-using-anaconda/) for the installation instruction)
* torchvision
* seaborn

## Project Details<a name="details"></a>

This project contains two notebooks which walk you through the experiment process where:
* [Analysis_Content_Style_Representation.ipynb](./Analysis_Content_Style_Representation.ipynb) - This notebook contains the Analysis of the Content Representation and Style Representations of the images that we will be using in this experiment, and I would highly suggest going over this notebook first to get a better understading of the concepts of the Style Transfer algorithm and the setup for the experiment.
   * In case there is an error when trying to render this notebook in github, you can download the html file [Analysis_Content_Style_Representation.html](./Analysis_Content_Style_Representation.html)
* [StyleTransfer_Experiment_Results.ipynb](./StyleTransfer_Experiment_Results.ipynb) - This notebook contains the Experiment results with various parameter settings
   * In case there is an error when trying to render this notebook in github, you can download the html file [StyleTransfer_Experiment_Results.html](./StyleTransfer_Experiment_Results.html)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

* Some part of the source code in this project was inspired by a [hands-on lab](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/style-transfer/Style_Transfer_Solution.ipynb) for the "Style Transfer" module of the [Udacity's Deep Learning Nanodegree Program](https://www.udacity.com/course/deep-learning-nanodegree--nd101) and must give credit to them.
* Images used in this experiment were downloaded online. Please find where those images were downloaded in [./images/README.md](./images/README.md) and must give credit to the owner of those images.

This project is [MIT licensed](./LICENSE).