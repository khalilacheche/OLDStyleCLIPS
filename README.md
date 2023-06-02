
# OLDStyleCLIPS: Empowering Image Editing with Textual Guidance using StyleGAN Latent Space Optimization, CLIP, and Image Segmentation
Khalil Haroun Achache, Farah Briki, Haitao Zhou <br>


Optimization: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/khalilacheche/StyleCLIP/blob/main/notebooks/playground.ipynb)

Image Inversion: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khalilacheche/StyleCLIP/blob/main/notebooks/inversion_playground.ipynb)



## Description
This project aims at performing text based editing on StyleGAN2 images. It was done in the context of the course of Computational Photography (CS-413) at EPFL.
This project is based on the paper [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://arxiv.org/abs/2103.17249) by Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, Dani Lischinski, and Daniel Cohen and [LELSD: OPTIMIZING LATENT SPACE DIRECTIONS FOR GAN-BASED LOCAL IMAGE EDITING](https://arxiv.org/pdf/2111.12583.pdf) by Ehsan Pajouheshgar, Tong Zhang, and Sabine Süsstrunk.



### Repo Structure

- `notebooks/`: contains the notebooks used for the project
    - `notebooks/playground.ipynb`: the notebook to run on colab to perform image editing
    - `notebooks/inversion_playground.ipynb`: the notebook to run on colab to perform image inversion, i.e. find the latent code of an image
- `models/`: contains the wrappers for the models used in the project
- `utils.py`: contains the utility functions used in the project
- `licenses/`: contains the licenses of the models used in the project
- `img/`: contains the images for this README
- `optimization/`: contains the code for running the main method of the project
- `pretrained/`: contains the weights of some of the pretrained models used in the project (that are uploadable to github, the others are hosted on google drive)
- `criteria/`: contains wrappers for the different losses used for optimization
