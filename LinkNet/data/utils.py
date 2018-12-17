import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K


def get_files(folder, name_filter=None, extension_filter=None):
    """Returns the list of files in a folder.

    Args:
        folder (string): The path to a folder.
        name_filter (string, optional): The returned files must contain
            this substring in their filename. Default: None; files are not
            filtered.
        extension_filter (string, optional): The desired file extension.
            Default: None; files are not filtered.

    Returns:
        The list of files.

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def pil_loader(data_path, label_path, shape):
    """Loads a sample and label image given their path as PIL images.

    Args:
        data_path (string): The filepath to the image.
        label_path (string): The filepath to the ground-truth image.
        shape (tuple): The requested size in pixels, as a 2-tuple:
            (width,height). If set to ``None``, resizing is not performed.

    Returns:
        The image and the label as PIL images.

    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    if shape is not None:
        if data.size != shape:
            data = data.resize(shape)
        if label.size != shape:
            label = label.resize(shape)

    return data, label


def remap(image, old_values, new_values):
    """Replaces pixels values with new values.

    Pixel values from ``old_values`` in ``image`` are replaced index by
    index with values from ``new_values``.

    Args:
        image (numpy.ndarray): The image to process.
        old_values (tuple): A tuple of values to be replaced.
        new_values (tuple): A tuple of new values to replace ``old_values``.

    Returns:
        The image with remapped classes.

    """
    assert type(new_values) is tuple, "new_values must be of type tuple"
    assert type(old_values) is tuple, "old_values must be of type tuple"
    assert len(new_values) == len(old_values), (
        "new_values and old_values must have the same length"
    )

    # Images with more than one channel are assumed to be in categorical format
    # therefore, they are converted to integer format
    if image.shape[-1] > 1:
        image = np.argmax(image, axis=-1)

    # Replace old values by the new ones
    remapped_img = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values
        # equal to 0
        if new != 0:
            remapped_img[image == old] = new

    return remapped_img


def imshow_batch(image_batch, nrows=1, figsize=None):
    """Shows a batch of images in a grid.

    Note: Blocks execution until the figure is closed.

    Args:
        image_batch (numpy.ndarray): A batch of images. Dimension is assumed
            as (batch, height, width, channels); or, (height, width, channels)
            which is transformed into (1, height, width, channels).
        nrows (int): The number of rows of the image grid. The number of
            columns is infered from the rows and the batch size.
        figsize (tuple, optional): The size of the figure (width, height)
            in inches. Default: None (defaults to rc figure.figsize)

    """
    assert nrows > 0, "number of rows must be greater than 0"
    assert figsize is None or isinstance(
        figsize, tuple
    ), ("expect type None or tuple for figsize")

    if (np.ndim(image_batch) == 3):
        image_batch = np.expand_dims(image_batch, 0)

    # Compute the number of columns needed to plot the batch given the rows
    ncols = int(np.ceil(image_batch.shape[0] / nrows))

    # Show the images with subplot
    if figsize is None:
        figsize = plt.rcParams.get('figure.figsize')

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for idx in range(image_batch.shape[0]):
        if nrows == 1:
            axes[idx].imshow(image_batch[idx].astype(int))
        else:
            col = idx % ncols
            row = idx // ncols
            axes[row, col].imshow(image_batch[idx].astype(int))

    plt.show()


def categorical_to_rgb(categorical_batch, class_to_rgb):
    """Converts label(s) from categorical format to RGB representation.

    Args:
        categorical_batch (numpy.ndarray): A batch of labels in categorical
            format. Dimension is assumed as (batch, height, width, channels);
            or, (height, width, channels) which is transformed into
            (1, height, width, channels).
        class_to_rgb (OrderedDict): An ordered dictionary that relates pixel
            values, class names, and class colors.

    Returns:
        The label(s) as RGB images.

    """
    if (np.ndim(categorical_batch) == 3):
        categorical_batch = np.expand_dims(categorical_batch, 0)

    rgb_batch = np.zeros(
        (
            categorical_batch.shape[0],
            categorical_batch.shape[1],
            categorical_batch.shape[2],
            3,
        ),
        dtype=np.uint8
    )
    for idx, image in enumerate(categorical_batch):
        image = np.argmax(image, axis=-1).squeeze()
        for class_value, (class_name, rgb) in enumerate(class_to_rgb.items()):
            rgb_batch[idx][image == class_value] = rgb

    return rgb_batch


def rgb_to_categorical(image_batch, class_to_rgb):
    """Converts labels from RGB to categorical representation.

    Args:
        image_batch (numpy.ndarray): A batch of labels in the RGB color-space
            Dimension is assumed as (batch, height, width, channels);
            or, (height, width, channels) which is transformed into
            (1, height, width, channels).
        class_to_rgb (OrderedDict): An ordered dictionary that relates pixel
            values, class names, and class colors.

    Returns:
        The label(s) in categorical format.

    """
    if (np.ndim(image_batch) == 3):
        image_batch = np.expand_dims(image_batch, 0)

    categorical_batch = np.zeros(
        (
            image_batch.shape[0],
            image_batch.shape[1],
            image_batch.shape[2],
            len(class_to_rgb),
        ),
        dtype=np.uint8
    )
    for idx, image in enumerate(image_batch):
        for class_value, (class_name, rgb) in enumerate(class_to_rgb.items()):
            # Create mask of pixels that match the rgb code for this class
            mask = np.all(image == rgb, axis=-1)

            # Assign the one-hot vector representation of the class to the
            # categorical image inside the batch. The line below outputs the
            # following:
            # k = 1; M = 3 -> (0, 1, 0)
            # k = 2; M = 3 -> (0, 0, 1)
            onehot = np.eye(1, M=len(class_to_rgb), k=class_value).ravel()
            categorical_batch[idx][mask] = onehot

    return categorical_batch

def _mean_iou_loss(y_true, y_pred, num_classes=3):
    """Computes the mean intesection over union using numpy.

    Args:
        y_true (tensor): True labels.
        y_pred (tensor): Predictions of the same shape as y_true.

    Returns:
        The mean intersection over union (np.float32).

    """
    # Compute the confusion matrix to get the number of true positives,
    # false positives, and false negatives
    # Convert predictions and target from categorical to integer format
    target = np.argmax(y_true, axis=-1).ravel()
    predicted = np.argmax(y_pred, axis=-1).ravel()

    # Trick for bincounting 2 arrays together
    x = predicted + num_classes * target
    bincount_2d = np.bincount(
        x.astype(np.int32), minlength=num_classes**2
    )
    assert bincount_2d.size == num_classes**2
    conf = bincount_2d.reshape(
        (num_classes, num_classes)
    )

    # Compute the IoU and mean IoU from the confusion matrix
    true_positive = np.diag(conf)
    false_positive = np.sum(conf, 0) - true_positive
    false_negative = np.sum(conf, 1) - true_positive

    # Just in case we get a division by 0, ignore/hide the error and
    # set the value to 1 since we predicted 0 pixels for that class and
    # and the batch has 0 pixels for that same class
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / (true_positive + false_positive + false_negative)
    iou[np.isnan(iou)] = 1

    return 1-np.mean(iou).astype(np.float32)

def mean_iou_loss(y_true, y_pred):
    return tf.py_func(_mean_iou_loss, [y_true, y_pred], tf.float32)

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.flatten(K.argmax(y_true, axis=-1))
    pred_pixels = K.flatten(K.argmax(y_pred, axis=-1))
    void_labels = K.flatten(K.equal(K.sum(y_true, axis=-1), 0))
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        #legal_batches = K.sum(tf.to_int32(true_labels))>0
        ious = K.sum(inter)/K.sum(union)
        iou.append(ious)
        #iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return tf.to_float(K.mean(iou))

def IOU_loss(y_true, y_pred):
    return -K.log(tf.reshape(tf.metrics.mean_iou(labels=y_true, predictions=y_pred, num_classes=3),[]))
    #return tf.to_float(-K.log(Mean_IOU(y_true, y_pred) + K.epsilon()))


def soft_dice_coef_loss(y_pred, y_true):
    return 1 - soft_dice_coef(y_true, y_pred, epsilon=1e-6)

def soft_dice_coef(y_true, y_pred, epsilon=1e-6):
    #print("1: ", y_pred.shape)
    #print("2: ", len(y_pred.shape))
    #print("3: ", range(1, len(y_pred.shape)-1))
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    #print("axes: ", axes)
    numerator = 2. * K.sum(y_pred * y_true, axes)
    #print("y_pred * y_true: ", y_pred * y_true)
    #print("numerator: ", numerator)
    #print("numerator shape: ", numerator.shape)
    denominator = K.sum(np.square(y_pred) + np.square(y_true), axes)

    return (K.mean(numerator / (denominator + epsilon)))
