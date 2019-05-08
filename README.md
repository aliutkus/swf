# Sliced Wasserstein Flows

Authors pytorch implementation of [Sliced-Wasserstein Flows: Nonparametric Generative Modeling via Optimal Transport and Diffusions](https://arxiv.org/abs/1806.08141), presented at ICML 2019.

![I need to change this but it's too hard](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

If you find this code useful, thank you to cite the following work:
> @inproceedings{Liutkus2019SWF,  
  author    = {A. Liutkus and U. Simsekli and S. Majewski and A. Durmus and F.-R. St\"oter}  
  title     = {Sliced-Wasserstein Flows: Nonparametric Generative Modeling via Optimal Transport and Diffusions},  
  booktitle = {Proceedings of the 36th International Conference on Machine Learning,
               (ICML) 2019},  
  address = {Long Beach, CA, USA},  
  month =  jun,  
  year      = {2019},  
}

# Installation
This code depends on several packages. To use it, we suggest you use the conda package manager and follow the following steps after cloning:
1.  Create a `SWF` environment using the provided `environment-cpu.yml` or `environment-gpu.yml` files:
> conda env create -f environment-gpu.yml

  This will install through conda all the packages you need that do not need manual installation.
2. Activate the environment through `source activate SWF`
3. Due to the fact SWF uses some non-conventional quantile and interpolation operations, we had to implement some features that are not (yet?) available in pytorch. For this reason, you need to manually install some packages. Follow the installation instructions of the following repositories:
  * [Pytorch searchsorted](https://github.com/aliutkus/torchsearchsorted) for a CUDA implementation of the `searchsorted` function.
  * [Pytorch interp1d](https://github.com/aliutkus/torchinterp1d) for 1D-interpolation with pytorch.
  * [Pytorch percentile](https://github.com/aliutkus/torchpercentile) for an implementation of the `percentile` function in pytorch on GPU.
  * [qsketch](https://github.com/aliutkus/qsketch) as a toolbox for low-level routines related to Sliced Wasserstein training.

When this is done, you should be able to try out the code. For instance, run the demo by going in the `code` folder and type:
> ./demotoy.sh

This will start SWF with the toy example found in the paper and write out results in your `~/swf_samples_toy` folder.

# Options in `demo.sh`
To play around with SWF, we have prepared a more sophisticated demo script called `demo.sh`. You may want to play with its options. With the `SWF` environment active, the way to use it is to run:
> ./demo.sh DATASET

where supported values for `DATASET` are:
* Any torchvision dataset, e.g. MNIST
* `toy` in which case the toy GMM example described in the paper will be used
* `CelebA` in which case you need to manually download the dataset, after which the code should be able to handle that dataset.

Other options worth mentioning are the following:
* Sketching parameters
  * `NUM_SKETCHES`: int  
    The number of groups of random projections that are considered.
  * `NUM_THETAS`: int  
    The number of random projections per sketch. The total number of random projections considered is hence NUM_SKETCHES * NUM_THETAS
  * `NUM_QUANTILES`: int  
    The number of quantiles to consider
  * `NUM_EXAMPLES`: int  
    The number of examples to use from the data to compute each sketch.
* Images processing parameters
  * `IMG_SIZE`: int  
    In the case data are images (not toy), they will be interpolated as `IMG_SIZE x IMG_SIZE` pixels images.
  * `AE_STRING`: either "" or "--ae"  
    If equal to "--ae", an autoencoder will be trained to reduce the dimension. By default, the weights of this will be saved in the `weights` subfolder after training. (Training will be automatically launched if this configuration has not been trained before)
  * `CONV_AE_STRING`: either "" or "--conv_ae"  
    If equal to "--conv_ae", the autoencoder trained will be convolutive.
  * `BOTTLENECK_SIZE`: int  
    Number of bottleneck features (the actual dimension SWF will care about).
* SWF parameters
  * `STEPSIZE`: float  
    The stepsize for each step in the SWF, as in the paper
  * `REG`: float  
    The lambda parameter in the paper for additive noise at each step.
  * `NUM_EPOCH`: int  
    The number of iterations of SWF
  * `NO_FIXED_SKETCH_STRING`: either "" or "--no_fixed_sketch"  
    If equal to "--no_fixed_sketch", then different sketches (random projections) will be considered at each epoch. This is not the strategy described in the paper.
  * `NUM_SAMPLES`: int  
    The number of particles for SWF.
  * `INPUT_DIM`: int  
    The dimension for the initial particles. If `-1`, then the particles will have the same shape as the target data. If it's different, then the initial particles will be multiplied by a random matrix of appropriate size before calling SWF.
* Test particles parameters
  * `NUM_TEST`: int  
    The number of samples on which we must apply a pre-trained SWF.
  * `TEST_TYPE`: either "INTERPOLATE" or "RANDOM"  
    If "INTERPOLATE", then test particles will be picked as linear interpolations between successive train particles. If "RANDOM", then they will be picked as totally random from the same distribution.
* Plot parameters
  * `PLOT_EVERY`: int
    Will make the different plots every PLOT_EVERY epochs. -1 for no plots
  * `PLOT_NB_TRAIN`: int
    the number of train particles to plot
  * `PLOT_NB_TEST`: int
    the number of test particles to plot
  * `MATCH_EVERY`: int
    _warning_ matching is quite slow ! Will find the closest samples from the training data every MATCH_EVERY epochs.

# Additional information
## Main file: `swf.py`
This file contains the main `swf` function that operates the SWF. Its different parameters are:
* The two sets of particles (train and test)
* The basic parameters for the flow: stepsize, regularization, number of epochs
* A `Sketcher` and `ModulesDataset` objects, that are defined in the `qsketch` repository. These are handy ways of dealing with sketching: the sketcher has a `queue` member that is simply filled with new entries as they come. The ModulesDataset constructs new projections on demand. The particular generalization twist we have chosen is to actually allow for any operation on the data as a preprocessing for quantiles computation, not only linear projection. This is to allow some other further works on SW and generalized SW learning.

The `__main__` block just checks the parameters, constructs the data stream, sketcher, and launches the SWF.

## Plot stuff: `plotting.py`
This file contains lots of hacks and things to do the nice plotting. It basically consists of a class `SWFPlot`, that contains a `log` method. This is the one that is called by the SWF at each epoch with all local variables, and decides what to do.

## Data stuff: `data.py`
This file declares the convenient `TransformedDataset`, that allows to apply some function after accessing some elements from another dataset. In that sense, this is similar to _transforms_, but is useful when transforms cannot be used. This is the case here with the AE strategy: we want to access some datasets, but the items are actually the bottleneck features after encoding.

Additionally, this file allows to load data from various sources, including torchvision datasets, but also CelebA and the toy example.

## `networks`: some implementations of nets
This folder contains some basic dense and convolutional autoencoder of parameterized bottleneck size, as well as the definition of a LinearProjector module, that simply has
the particularity of providing a direct access to its backward operation (not needing a forward pass), because this is required for SWF.
