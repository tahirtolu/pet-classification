import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def define_paths(data_dir):
    """Generate data paths with labels"""
    print(f"Scanning directory: {data_dir}")
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    print(f"Found {len(folds)} folders")
    
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if os.path.isdir(foldpath):
            filelist = os.listdir(foldpath)
            print(f"Found {len(filelist)} files in {fold}")
            for file in filelist:
                fpath = os.path.join(foldpath, file)
                filepaths.append(fpath)
                labels.append(fold)

    print(f"Total files found: {len(filepaths)}")
    return filepaths, labels

def define_df(files, classes):
    """Concatenate data paths with labels into one dataframe"""
    print("Creating dataframe...")
    Fseries = pd.Series(files, name='filepaths')
    Lseries = pd.Series(classes, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    print(f"Dataframe created with {len(df)} rows")
    return df

def create_df(tr_dir, val_dir, ts_dir):
    """Create dataframes for train, validation and test sets"""
    print("\nCreating dataframes for all sets...")
    
    # train dataframe
    print("\nProcessing training data...")
    files, classes = define_paths(tr_dir)
    train_df = define_df(files, classes)

    # validation dataframe
    print("\nProcessing validation data...")
    files, classes = define_paths(val_dir)
    valid_df = define_df(files, classes)

    # test dataframe
    print("\nProcessing test data...")
    files, classes = define_paths(ts_dir)
    test_df = define_df(files, classes)

    return train_df, valid_df, test_df

def create_gens(train_df, valid_df, test_df, batch_size):
    """Create image data generators for train, validation and test sets"""
    print("\nCreating data generators...")
    
    # define model parameters
    img_size = (224, 224)
    channels = 3
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # Calculate test batch size
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) 
                                 if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size
    print(f"Test batch size: {test_batch_size}, Test steps: {test_steps}")

    # Data augmentation for training
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function=scalar, 
                              horizontal_flip=True)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    # Create generators
    print("\nCreating training generator...")
    train_gen = tr_gen.flow_from_dataframe(
        train_df, 
        x_col='filepaths', 
        y_col='labels', 
        target_size=img_size, 
        class_mode='categorical',
        color_mode=color, 
        shuffle=True, 
        batch_size=batch_size
    )
    print(f"Training classes: {train_gen.class_indices}")

    print("\nCreating validation generator...")
    valid_gen = ts_gen.flow_from_dataframe(
        valid_df, 
        x_col='filepaths', 
        y_col='labels', 
        target_size=img_size, 
        class_mode='categorical',
        color_mode=color, 
        shuffle=True, 
        batch_size=batch_size
    )
    print(f"Validation classes: {valid_gen.class_indices}")

    print("\nCreating test generator...")
    test_gen = ts_gen.flow_from_dataframe(
        test_df, 
        x_col='filepaths', 
        y_col='labels', 
        target_size=img_size, 
        class_mode='categorical',
        color_mode=color, 
        shuffle=False, 
        batch_size=test_batch_size
    )
    print(f"Test classes: {test_gen.class_indices}")

    return train_gen, valid_gen, test_gen

def show_images(gen):
    """Display sample images from the generator"""
    print("\nDisplaying sample images...")
    g_dict = gen.class_indices
    classes = list(g_dict.keys())
    images, labels = next(gen)

    length = len(labels)
    sample = min(length, 25)

    plt.figure(figsize=(20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()

def plot_label_count(df, plot_title):
    """Plot label counts in the dataset"""
    print(f"\nPlotting label counts for {plot_title} data...")
    vcounts = df['labels'].value_counts()
    labels = vcounts.keys().tolist()
    values = vcounts.tolist()
    lcount = len(labels)

    if lcount > 55:
        print('The number of labels is > 55, no plot will be produced')
    else:
        plot_labels(lcount, labels, values, plot_title)

def plot_labels(lcount, labels, values, plot_title):
    """Helper function to plot labels"""
    width = lcount * 4
    width = np.min([width, 20])

    plt.figure(figsize=(width, 5))

    form = {'family': 'serif', 'color': 'blue', 'size': 25}
    sns.barplot(x=labels, y=values)
    plt.title(f'Images per Label in {plot_title} data', fontsize=24, color='blue')
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('CLASS', fontdict=form)
    plt.ylabel('IMAGE COUNT', fontdict=form)

    rotation = 'vertical' if lcount >= 8 else 'horizontal'
    for i in range(lcount):
        plt.text(i, values[i] / 2, str(values[i]), fontsize=12,
                rotation=rotation, color='yellow', ha='center')

    plt.show() 