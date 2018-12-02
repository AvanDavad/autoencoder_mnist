import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import subprocess
import imageio
import string
import os
from cons import IMG_H, IMG_W

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def _transform(x):
    x = x.astype(np.float32)/255.
    x *= 0.8
    x +=0.1
    x = x[...,np.newaxis]
    return x

x_train = _transform(x_train) # (60000,28,28,1)
x_test  = _transform(x_test) # (10000,28,28,1)
x_all = np.concatenate([x_train, x_test], axis=0) # (70000,28,28,1)
y_all = np.concatenate([y_train, y_test], axis=0) # (70000,)

def get_train_data():
    return (x_train, y_train)

def get_test_data():
    return (x_test, y_test)

def plot_img(batch, idx):
    img = batch[idx,:,:,0]
    ax = plt.subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(img, cmap=plt.get_cmap('binary'))
    plt.show()
    
def get_random_batch(x, size):
    idx = np.random.choice(len(x), size=size, replace=False)
    return x[idx]

def get_random_batch_with_labels(x, y, size):
    idx = np.random.choice(len(x), size=size, replace=False)
    return x[idx], y[idx]    

def plot_images(imgs):
    idx = np.random.choice(len(imgs),size=9,replace=False)
    plt.figure(figsize=(12,8))
    for i in range(9):
        ax = plt.subplot(331+i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(imgs[idx[i],:,:,0], cmap=plt.get_cmap('binary'))
    plt.show()
    
def distplot(x):
    x = x.flatten()
    sns.distplot(x, kde=False)

def get_images_from_generated(gen, step_size):
    images = []
    gen = np.clip((gen - 0.1) / 0.8, 0., 1.)
    gen = 1. - gen
    for idx in range(0,len(gen),step_size):
        img = np.zeros([IMG_H,IMG_W,3])
        img += gen[idx,:,:,0:1]*255.
        img = img.astype(np.uint8)
        images.append(img)
    return images
    
def get_temp_dir():
    tmpdir = _generate_temp_dir()
    while os.path.isdir(tmpdir):
        tmpdir = _generate_temp_dir()
    os.mkdir(tmpdir)
    return tmpdir

def _generate_temp_dir():
    return 'temp_'+''.join(np.random.choice(list(string.printable[:62]), size=20, replace=True))

def export_images(images, name='images.gif', delay=5):
    tmpdir = get_temp_dir()
    for i, img in enumerate(images):
        imageio.imwrite('{}/{:03}.png'.format(tmpdir,i), images[i])
    subprocess.call('convert -delay {} -loop 0 {}/*.png {}'.format(delay,tmpdir,name), shell=True)
    subprocess.call('rm {}/*'.format(tmpdir), shell=True)
    os.rmdir(tmpdir)
    print('{} was created'.format(name))

def plot_from_code(code, model, model_path):
    input_code = np.zeros([model.batch_size, model.code_size]).astype(np.float32)
    input_code[0] = np.array(code)
    _, gen = model.feedforward(None, input_code=input_code, model_path=model_path)
    plot_img(gen, 0)

def get_batch_with_label(y, batch_size):
    idxs = np.random.choice(np.where(y_all==y)[0], size=batch_size, replace=False)
    batch = x_all[idxs]
    return batch
