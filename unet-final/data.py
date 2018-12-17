from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.utils import np_utils
import math

# masks mapping dictionary. 0: Background, 127: False Lumen, 254: True Lumen
mask_dict = np.array([0, 127, 254])

# This method is used to measure Dice Score during
# test time. 
#     - y_pred (model output)
#     - y_true (ground truth)
def soft_dice_coef(y_pred, y_true , epsilon=1e-6): 
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    return (np.mean(numerator / (denominator + epsilon)))

# This method is used to measure Dice Loss based on
# measure Dice Score during test time
#     - y_pred (model output)
#     - y_true (ground truth)
def soft_dice_coef_loss(y_pred, y_true):
	return 1 - soft_dice_coef(y_true, y_pred, epsilon=1e-6)

# This method is used to store the predicted output masks by model to file system
#     - save_path (path to store generated results)
#     - results (predicted output by model)
def save_result(save_path, results):
    for i in range(results.shape[0]):
        img = results[i]
        img_out = np.zeros((img.shape[0],img.shape[1]))
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                c = np.argmax(img[j][k])
                img_out[j][k] = mask_dict[c]
        img_out_1= np.array(img_out, dtype=np.uint8)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_out_1)

# This method is used to store original masks images used in testing
#     - y_test (original masks used in testing for comparison to prediction)
def save_label_3d (y_test):
	p = 5
	d = y_test.shape[0]/p
	for i in range(y_test.shape[0]):
		img = y_test[i]
		img_reshape = img.reshape([4,256,256,3])
		for z in range(img_reshape.shape[0]):
			img1 = img_reshape[z] 
			img_out = np.zeros((img1.shape[0],img1.shape[1]))
			for j in range(img1.shape[0]):
				for k in range(img1.shape[1]):
					c = np.argmax(img1[j][k])
					img_out[j][k] = mask_dict[c]
			b = math.floor(i/d)
			r = i%d
			img_out_1= np.array(img_out, dtype=np.uint8)
			io.imsave(os.path.join("data/test_mask","%d_mask.png"%((800*b)+(8*r)+(z*2))),img_out_1)

# This method is used to store the predicted output masks by model to file system
# for 3D model
#     - save_path (path to store generated results)
#     - results (predicted output by model)
def save_result_3d (save_path, results):
    p = 5
    d = results.shape[0]/p
    for i in range(results.shape[0]):
        img = results[i]
        img_reshape = img.reshape([4,256,256,3])
        for z in range(img_reshape.shape[0]):
            img = img_reshape[z] 
            img_out = np.zeros((img.shape[0],img.shape[1]))
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    c = np.argmax(img[j][k])
                    img_out[j][k] = mask_dict[c]
            b = math.floor(i/d)
            r = i%d
            img_out_1= np.array(img_out, dtype=np.uint8)
            io.imsave(os.path.join(save_path,"%d_predict.png"%((800*b)+(8*r)+(z*2))),img_out_1)

# This method is used to get images from original dataset
# to be used for training in 3D model. Returned images are normalized  
# and tranformed to from 256x256 to 256x256x1 to make it 
# possible to be used in Keras model
# Note: since there is huge similarity between sequential images and
# due to memory limitation the whole study was sample every at every 
# 5th images and then each 4 sequential images of sampled data 
# formed one input of shape 4x256x256x1
def train_all_3d_new(as_gray = True):
	images = []
	train_path = "data/train/train_all"
	for z in range(1, 25):
		for i in range(0, 760, 20):
			imd3d = []
			
			for j in range (0,20,5):
				img1 = io.imread(os.path.join(train_path,"%d_train.jpg"%(((z-1)*800)+i+j)),as_gray = as_gray)
				img1 = img1 / 255
				img1 = np.reshape(img1,img1.shape+(1,))
				imd3d.append(img1)
			
			images.append(np.array(imd3d))			    
	return(np.array(images))

# This method is used to get masks from original dataset
# to be used for training in 3D model. Method also transform masks
# to one-hot encoded version result in 256x256x3 images
# Note: since there is huge similarity between sequential images and
# due to memory limitation the whole study was sampled every at every 
# 5th images and then each 4 sequential images of sampled data 
# formed one mask of shape 4x256x256x3
def label_all_3d_new(as_gray = True):
	labels = []
	mask_path = "data/mask/mask_all"
	for z in range(1, 25):
		for i in range(0, 760, 20):
			mask3d = []
			
			for j in range (0,20,5):
				mask1 = io.imread(os.path.join(mask_path,"%d_mask.png"%(((z-1)*800)+i+j)),as_gray = as_gray)
				new_mask1 = np.zeros(mask1.shape + (3,))
				mask_val1 = np.array([0, 127, 254])
				for k in range(3):
					new_mask1[mask1 == mask_val1[k],k] = 1
				mask3d.append(new_mask1)
			
			labels.append(np.array(mask3d))			    
	return(np.array(labels))

# This method is used to get images from original dataset
# to be used for training. Returned images are normalized  
# and tranformed to from 256x256 to 256x256x1 to make it 
# possible to be used in Keras model
def train_all_new(as_gray = True):
	images = []
	train_path_img = 'data/train/train_all'
	print(train_path_img)
	for i in range(1, 19200, 2):
		img = io.imread(os.path.join(train_path_img,"%d_train.jpg"%i),as_gray = as_gray)
		img = img / 255
		img = np.reshape(img,img.shape+(1,))
		images.append(img)			    
	return(np.array(images))

# This method is used to get masks from original dataset
# to be used for training. Method also transform masks
# to one-hot encoded version result in 256x256x3 images
def label_all_new(as_gray = True):
	labels = []
	mask_path_img = 'data/mask/mask_all'
	print(mask_path_img)
	for i in range(1, 19200, 2):
		mask = io.imread(os.path.join(mask_path_img,"%d_mask.png"%i),as_gray = as_gray)
		new_mask = np.zeros(mask.shape + (3,))
		mask_val = np.array([0, 127, 254])
		for k in range(3):
			new_mask[mask == mask_val[k],k] = 1
		labels.append(new_mask)			    
	return(np.array(labels))

# This method is used to get images from new dataset
# used in generalized testing. Returned images are 
# normalized and tranformed to from 256x256 to 256x256x1
# to make it possible to be used in Keras model 
def test_train_new(as_gray = True):
	images = []
	for k in range(1, 6):
		train_path = 'data/pt' + "%d"%k
		train_path_img = train_path + '/images'
		print(train_path_img)
		for j in range(0, 800, 2):
			img = io.imread(os.path.join(train_path_img,"%d.jpg"%j),as_gray = as_gray)
			img = img / 255
			img = np.reshape(img,img.shape+(1,))
			images.append(img)		
	return(np.array(images))

# This method is used to get masks from new dataset used
# used in generalized testing. Method also transforms masks
# to one-hot encoded version result in 256x256x3 images
def test_label_new(as_gray = True):
	labels = []
	for t in range(1, 6):
		mask_path = 'data/pt' + "%d"%t
		mask_path_img = mask_path + '/aorta_masks'
		print(mask_path_img)
		for j in range(0, 800, 2):
			mask = io.imread(os.path.join(mask_path_img,"%d.png"%j),as_gray = as_gray)
			new_mask = np.zeros(mask.shape + (3,))
			mask_val = np.array([0, 127, 254])
			for k in range(3):
				new_mask[mask == mask_val[k],k] = 1
			labels.append(new_mask)			 
	return(np.array(labels))

# This method is used to get images from new dataset
# used in generalized testing for 3D model. Returned images are 
# normalized and tranformed to from 256x256 to 256x256x1
# to make it possible to be used in Keras model. 
# 3D model images are generated by grabing 4 sequential axial images  
# Note: since there is huge similarity between sequential images and
# due to memory limitation the whole study was sample every at every 
# 5th images and then each 4 sequential images of sampled data 
# formed one input of shape 4x256x256x1
def test_train_new_3d(as_gray = True):
	images = []
	for k in range(1, 6):
		train_path = 'data/pt' + "%d"%k
		train_path_img = train_path + '/images'
		for i in range(17, 700, 100):
			imd3d = []
			
			for j in range (0,20,5):
				img1 = io.imread(os.path.join(train_path_img,"%d.jpg"%(i+j)),as_gray = as_gray)
				img1 = img1 / 255
				img1 = np.reshape(img1,img1.shape+(1,))
				imd3d.append(img1)
			
			images.append(np.array(imd3d))			    
	return(np.array(images))

# This method is used to get masks from new dataset used
# used in generalized testing for 3D model. Method also transforms 
# masks to one-hot encoded version result in 256x256x3 images.
# 3D model masks are generated by grabing 4 sequential axial images  
# Note: since there is huge similarity between sequential images and
# due to memory limitation the whole study was sampled every at every 
# 5th images and then each 4 sequential images of sampled data 
# formed one mask of shape 4x256x256x3
def test_label_new_3d(as_gray = True):
	labels = []
	for t in range(1, 6):
		mask_path = 'data/pt' + "%d"%t
		mask_path_img = mask_path + '/aorta_masks'
		for i in range(17, 700, 100):
			mask3d = []
			
			for j in range (0,20,5):
				mask1 = io.imread(os.path.join(mask_path_img,"%d.png"%(i+j)),as_gray = as_gray)
				new_mask1 = np.zeros(mask1.shape + (3,))
				mask_val1 = np.array([0, 127, 254])
				for k in range(3):
					new_mask1[mask1 == mask_val1[k],k] = 1
				mask3d.append(new_mask1)
			
			labels.append(np.array(mask3d))			    
	return(np.array(labels))

# This method is used to get images from dataset that is
# shared between linknet and unet for training. Returned images are 
# normalized and tranformed to from 256x256 to 256x256x1
# to make it possible to be used in Keras model. 
# Note: common dataset is used to be able to compare the results of
# two different model
def shared_get_train(as_gray = True):
	images = []
	train_path = "data/indra_train"
	for z in range(0, 1000):
		img = io.imread(os.path.join(train_path,"%d_train.jpg"%z),as_gray = as_gray)
		img = img / 255
		img = np.reshape(img,img.shape+(1,))
		images.append(img)
	return(np.array(images))

# This method is used to get masks from dataset that is
# shared between linknet and unet for training. Method also transforms 
# masks to one-hot encoded version result in 256x256x3 images.
# Note: common dataset is used to be able to compare the results of
# two different model
def shared_get_mask(as_gray = True):
	labels = []
	mask_path = "data/indra_mask_train"
	for z in range(0, 1000):
		mask = io.imread(os.path.join(mask_path,"%d_mask.png"%z),as_gray = as_gray)
		new_mask = np.zeros(mask.shape + (3,))
		mask_val = np.array([0, 127, 254])
		for k in range(3):
			new_mask[mask == mask_val[k],k] = 1
		labels.append(new_mask)	
	return(np.array(labels))

# This method is used to get images from dataset that is
# shared between linknet and unet for testing. Returned images are 
# normalized and tranformed to from 256x256 to 256x256x1
# to make it possible to be used in Keras model. 
# Note: common dataset is used to be able to compare the results of
# two different model
def shared_get_train_test(as_gray = True):
	images = []
	train_path = "data/indra_test"
	for z in range(0, 200):
		img = io.imread(os.path.join(train_path,"%d_train.jpg"%z),as_gray = as_gray)
		img = img / 255
		img = np.reshape(img,img.shape+(1,))
		images.append(img)
	return(np.array(images))

# This method is used to get masks from dataset that is
# shared between linknet and unet for testing. Method also transforms 
# masks to one-hot encoded version result in 256x256x3 images.
# Note: common dataset is used to be able to compare the results of
# two different model
def shared_get_mask_test(as_gray = True):
	labels = []
	mask_path = "data/indra_mask_test"
	for z in range(0, 200):
		mask = io.imread(os.path.join(mask_path,"%d_mask.png"%z),as_gray = as_gray)
		new_mask = np.zeros(mask.shape + (3,))
		mask_val = np.array([0, 127, 254])
		for k in range(3):
			new_mask[mask == mask_val[k],k] = 1
		labels.append(new_mask)	
	return(np.array(labels))

# This method is used to get images from dataset that is
# shared between linknet and unet for validation. Returned images are 
# normalized and tranformed to from 256x256 to 256x256x1
# to make it possible to be used in Keras model. 
# Note: common dataset is used to be able to compare the results of
# two different model
def shared_get_train_dev(as_gray = True):
	images = []
	train_path = "data/indra_dev"
	for z in range(0, 200):
		img = io.imread(os.path.join(train_path,"%d_train.jpg"%z),as_gray = as_gray)
		img = img / 255
		img = np.reshape(img,img.shape+(1,))
		images.append(img)
	return(np.array(images))

# This method is used to get masks from dataset that is
# shared between linknet and unet for validation. Method also transforms 
# masks to one-hot encoded version result in 256x256x3 images.
# Note: common dataset is used to be able to compare the results of
# two different model
def shared_get_mask_dev(as_gray = True):
	labels = []
	mask_path = "data/indra_mask_dev"
	for z in range(0, 200):
		mask = io.imread(os.path.join(mask_path,"%d_mask.png"%z),as_gray = as_gray)
		new_mask = np.zeros(mask.shape + (3,))
		mask_val = np.array([0, 127, 254])
		for k in range(3):
			new_mask[mask == mask_val[k],k] = 1
		labels.append(new_mask)	
	return(np.array(labels))
