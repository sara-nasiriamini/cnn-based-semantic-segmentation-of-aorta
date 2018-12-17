# Keras-LinkNet [CS230-Indrasen Bhattacharya]

Keras implementation adapted from Keras-LinkNet developed by David Silva (davidtvs): https://github.com/davidtvs/Keras-LinkNet/
Readme from the original GitHub is appended below. Key changes made for the purpose of the CS230 project are summarized below.

## Changes for CS 230 Project
1. Additional DataGenerator for the aortic dissection dataset: dissection_generator.py, under the data directory
2. Modified decoder block in linknet.py under the models directory. The function decoder_block was modified: a dropout layer was added after each layer, and L2 regularization was also included. This was done in order to reduce overfitting and improve generalization accuracy (test accuracy on novel studies).
3. Added soft_dice_coef and soft_dice_coef_loss to utils.py (under data directory)

## Data processing scripts
The following python notebooks were used to transform the images to a format more suitable for learning (collected together in the folder 'data processing'):
1. convertRGB.ipynb: obtains png images and replicates data across channels (for both images and masks). Note that the images are downsampled from [256, 256] to [100, 100] - the downsampling speeded up training by a factor of 3x and had no effect on accuracy, since the relevant features are still visible in the downsampled images.
2. convertRGB_v2.ipynb: obtains png images for each study and constructs a polynomial fit across Z slices, using the 10 images before and after. The 3 coefficients (order 0, order 1 and order 2) are used as channels on the images. This eliminates local slice variation which is not expected to have a major effect on the segmentation, and may in fact add noise. The masks are copied and replicated (corresponding to the current Z-slice).
3. generateDataset.ipynb: Samples 1400 files from the entire dataset and splits them into training/validation/test after randomly permuting the list.
4. radonDisplay.ipynb: Generates and displays the radon transform of an image. This could be used to add tomographic reconstruction noise, for data augmentation in future implementation.

Other simple naming and file conversion operations were performed with shell commands.

## Checkpoints
The checkpoints contain weights for some of the trained LinkNets. A systematic hyperparameter search for learning rate was performed, followed with some exploration of dropout and L2 coefficients. The results are summarized in the report. The network weights are not included in the submission due to file size constraints.


## Readme from github source

Keras implementation of [*LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation*](https://arxiv.org/abs/1707.03718), ported from the lua-torch ([LinkNet](https://github.com/e-lab/LinkNet)) and PyTorch ([pytorch-linknet](https://github.com/e-lab/pytorch-linknet)) implementation, both created by the authors.

|                                Dataset                               | Classes <sup>1</sup> | Input resolution | Batch size | Mean IoU (%) |
|:--------------------------------------------------------------------:|:--------------------:|:----------------:|:----------:|:------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          12          |      960x480     |      2     |     47.15<sup>2</sup>    |
|           [Cityscapes](https://www.cityscapes-dataset.com/)          |          20          |     1024x512     |      2     |     53.37<sup>3</sup>    |

<sup>1</sup> Includes the unlabeled/void class.<br/>
<sup>2</sup> Test set.<br/>
<sup>3</sup> Validation set.

## Installation

1. Python 3 and pip.
2. Set up a virtual environment (optional, but recommended).
3. Install dependencies using pip: ``pip install -r requirements.txt``.


## Usage

Run [``main.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/main.py), the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [-h] [--mode {train,test,full}] [--resume]
               [--initial-epoch INITIAL_EPOCH] [--no-pretrained-encoder]
               [--weights-path WEIGHTS_PATH] [--batch-size BATCH_SIZE]
               [--epochs EPOCHS] [--learning-rate LEARNING_RATE]
               [--lr-decay LR_DECAY] [--lr-decay-epochs LR_DECAY_EPOCHS]
               [--dataset {camvid,cityscapes}] [--dataset-dir DATASET_DIR]
               [--workers WORKERS] [--verbose {0,1,2}] [--name NAME]
               [--checkpoint-dir CHECKPOINT_DIR]
```

For help on the optional arguments run: ``python main.py -h``


### Examples: Training

```
python main.py -m train --checkpoint-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


### Examples: Resuming training

```
python main.py -m train --resume True --initial-epoch 10 --checkpoint-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


### Examples: Testing

```
python main.py -m test --checkpoint-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


## Project structure

### Folders

- [``data``](https://github.com/davidtvs/Keras-LinkNet/tree/master/data): Contains code to load the supported datasets.
- [``metrics``](https://github.com/davidtvs/Keras-LinkNet/tree/master/metric): Evaluation-related metrics.
- [``models``](https://github.com/davidtvs/Keras-LinkNet/tree/master/models): LinkNet model definition.
- [``checkpoints``](https://github.com/davidtvs/Keras-LinkNet/tree/master/checkpoints): By default, ``main.py`` will save models in this folder. The pre-trained encoder (ResNet18) trained on ImageNet can be found here.

### Files

- [``args.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/arg.py): Contains all command-line options.
- [``main.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/main.py): Main script file used for training and/or testing the model.
- [``callbacks.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/callbacks.py): Custom callbacks are defined here.
