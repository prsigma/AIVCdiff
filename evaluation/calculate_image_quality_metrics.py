#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:02:08 2022

folder structure:
img_ref/
----img1.png
----img2.png
----img3.png
----....

img_gen/
----method1/
--------img1.png
--------img2.png
--------img3.png
--------....
----method2/
--------img1.png
--------img2.png
--------img3.png
--------....
----....

"""

import os
import torch_fidelity
from collections import OrderedDict
import datetime
import pandas as pd
import argparse
import torch
import random
import cv2
import torchvision.io as io
import numpy as np
join = os.path.join


def str2bool(v):
    """Convert string to boolean.

    Args:
        v (str): string to convert to boolean
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'TRUE', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'FALSE', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    return


def set_seed(seed):
    """Set seed for reproducibility.

    Args:
        seed (int): seed for reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return


def process_image(image_path, output_path):
    """Process image by rotating, flipping, or leaving it unchanged.
    
    Args:
        image_path (str): path to the image
        output_path (str): path to save the processed image
    """
    image = cv2.imread(image_path)
    action = random.choice(['rotate', 'flip', 'unchanged'])

    if action == 'rotate':
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif action == 'flip':
        direction = random.choice([-1, 0, 1])
        image = cv2.flip(image, direction)
    cv2.imwrite(output_path, image)
    return


def augment_imgs(directory, img_aug_count=500):
    """randomly copy and paste images in the directory to make
    it img_aug_count images.

    Args:
        directory (str): path to the directory containing images
    """
    import random
    imgs = os.listdir(directory)
    if len(imgs) > 0:
        while len(imgs) < img_aug_count:
            img = random.choice(imgs)
            img = img.strip("'")
            input_path = directory+'/'+img
            new_img_name = img[:-4].split('-copy')[0]+'-copy' +\
                str(len(imgs))+'.png'
            output_path = directory+'/'+new_img_name
            process_image(input_path, output_path)
            imgs = os.listdir(directory)
    return


def calculate_metrics(
        ref_path, gen_path, result_path, experiment,
        gen_pert, ref_pert, model):
    """Calculate FID and KID for the generated images.

    Args:
        ref_path (str): path to the reference images
        gen_path (str): path to the generated images
        result_path (str): path to save the result
        experiment (str): experiment name
        gen_pert (str): generated perturbation name
        ref_pert (str): reference perturbation name
        model (str): model name
    """
    print('Calculating metrics for '+model+' '+gen_pert+' '+ref_pert)
    print('Reference path: ', ref_path)
    print('Model generated path: ', gen_path)
    
    metric = OrderedDict()
    metric['experiment'] = []
    metric['generated_pert'] = []
    metric['reference_pert'] = []
    metric['model'] = []
    metric['FID'] = []
    metric['KID_mean'] = []
    metric['KID_std'] = []
    metric['seed'] = []

    seed = 42
    set_seed(seed)
    empty_file_counts = 0

    start = datetime.datetime.now()
    ref_imgs = []
    model_imgs = []
    
    for img in os.listdir(ref_path):
        img_path = join(ref_path, img)
        if os.path.getsize(img_path) == 0:
            empty_file_counts += 1
            print('Empty file: ', empty_file_counts)
            # remove the file
            os.remove(img_path)
            continue
        img = io.read_image(img_path)
        ref_imgs.append(img)
    ref_imgs = torch.stack(ref_imgs).float().cuda()

    for img in os.listdir(gen_path):
        img_path = join(gen_path, img)
        # check if file is empty
        if os.path.getsize(img_path) == 0:
            empty_file_counts += 1
            print('Empty file: ', empty_file_counts)
            os.remove(img_path)
            continue
        img = io.read_image(img_path)
        model_imgs.append(img)
    model_imgs = torch.stack(model_imgs).float().cuda()

    # check if there is any nan or infinity in model_imgs and ref_imgs
    if torch.isnan(model_imgs).any() or torch.isnan(ref_imgs).any():
        print('There is nan in the images')
        return

    if ref_imgs.shape != model_imgs.shape:
        min_shape = min(ref_imgs.shape[0], model_imgs.shape[0])
        ref_imgs = ref_imgs[:min_shape]
        model_imgs = model_imgs[:min_shape]

    # calculate FID, KID
    set_seed(seed)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=gen_path,
        input2=ref_path,
        cuda=True if torch.cuda.is_available() else False,
        fid=True,
        kid=True,
        batch_size=256,
        kid_subset_size=500,
        verbose=True,
    )

    metric['experiment'].append(experiment)
    metric['generated_pert'].append(gen_pert)
    metric['reference_pert'].append(ref_pert)
    metric['model'].append(model)
    metric['FID'].append(metrics_dict['frechet_inception_distance'])
    metric['KID_mean'].append(
        metrics_dict['kernel_inception_distance_mean'])
    metric['KID_std'].append(
        metrics_dict['kernel_inception_distance_std'])
    metric['seed'].append(seed)
    metric_df = pd.DataFrame(metric)
    
    print('Successfully calculated metrics for '+model+' '+gen_pert+' '+ref_pert)
    print('--'*50)
    print()

    with open(result_path, 'a') as f:
        metric_df.to_csv(f, header=f.tell() == 0, index=False)
    return


def evaluate_generated_images_per_pert(
        generated_pert, ref_img_path, gen_img_path, img_aug_count, experiment,
        model_list, result_path):
    """Calculate FID and KID for generated images.

    Args:
        generated_pert (str): perturbation name
        ref_img_path (str): path to the reference images
        gen_img_path (str): path to the generated images
        img_aug_count (int): number of images to be augmented
        experiment (str): experiment name
        model_list (list): list of models to be benchmarked
        result_path (str): path to save the result
    """

    # set up and check reference image directory for pert
    ref_img_path = join(ref_img_path, generated_pert)
    if not os.path.exists(ref_img_path):
        print('Reference image path for '+generated_pert+' does not exist')
        return

    if len(os.listdir(ref_img_path)) < img_aug_count:
        print('Augmenting reference images for '+generated_pert)
        augment_imgs(ref_img_path, img_aug_count)

    # set up result path and check if current evaluation is performed before
    if os.path.exists(result_path):
        result_df = pd.read_csv(result_path)

    for model in model_list:

        tmp_df = result_df[
            (result_df['generated_pert'] == generated_pert) &
            (result_df['reference_pert'] == generated_pert) &
            (result_df['model'] == model)]

        if not tmp_df.empty:
            print('Metrics for '+generated_pert+' and '+model+' already calculated')
            print('--'*50)
            print()
            continue

        if 'naive' in model.lower():
            model_gen_path = join(gen_img_path, model)
        else:
            model_gen_path = join(gen_img_path, generated_pert, model)

        if len(os.listdir(model_gen_path)) < img_aug_count:
            print('Number of images in '+model_gen_path+' is equal to '+str(len(os.listdir(model_gen_path)))+', which is less than '+str(img_aug_count))
            print('Augmenting predicted images for '+generated_pert+' '+model)
            augment_imgs(model_gen_path, img_aug_count)

        calculate_metrics(
            ref_img_path, model_gen_path, result_path, experiment,
            generated_pert, generated_pert, model)
        
    return


def create_result_file(args):
    """Create a result file to save the image quality metrics.

    Args:
        args (argparse): arguments
        across_pert (bool): indicating if the image quality metrics to be
            calculated per perturbation, or across perturbations.

    Returns:
        str: path to the result file
    """
        
    result_file_name = args.experiment+'_ood'+str(args.ood)
    if args.per_pert:
        result_file_name = result_file_name+'_per_pert'
    else:
        result_file_name = result_file_name+'_across_pert'

    result_file_name = result_file_name+'_img_quality_result.csv'

    result_path = args.result_path+'evaluation/' +\
        args.experiment+'/img_quality_result/'+result_file_name

    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            f.write('experiment,generated_pert,reference_pert,model,FID,' +
                    'KID_mean,KID_std,seed\n')

    return result_path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment',
                        default="",
                        help="experiment name")
    parser.add_argument('--img_aug_count',
                        default=1000,
                        help="number of images to be augmented")
    parser.add_argument('--ref_img_path',
                        default='',
                        help="directory address of real images")
    parser.add_argument('--gen_img_path',
                        default='',
                        help="directory address of the generated images")
    parser.add_argument('--result_path',
                        default="result/",
                        help="path to save test results")
    parser.add_argument('--model_file',
                        default="",
                        help="path to the file of models to be benchmarked")
    parser.add_argument('--pert_file_con',
                        default="",
                        help="path to the file of perturbations to be " +
                        "benchmarked for conditional model")
    parser.add_argument('--pert_file_naive', default="",
                        help="path to the file of perturbations to be " +
                        "benchmarked for naive model")
    parser.add_argument("--per_pert", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="indicating if image quality metric to be " +
                        "calculated per perturbation (between real " +
                        "and generated images) or between generated " +
                        "images and real images of other perturbations")
    parser.add_argument("--ood", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="indicating if perturbation list if from " +
                        "ood pert or in distribution list")
    parser.add_argument('--perturbation_subset_count',
                        default=0,
                        help="if not 0, only a subset of perturbations " +
                        "will be used for benchmarking")

    args = parser.parse_args()
    args.img_aug_count = int(args.img_aug_count)

    file_name = ''
    if args.ood:
        file_name = 'ood_pert_generated.txt'
    else:
        file_name = 'in_dist_pert_generated.txt'

    # get the list of models to be benchmarked
    model_df = pd.read_csv(args.model_file)
    model_list = model_df['model'].tolist()

    # get the list of perturbations to be benchmarked
    con_pert = pd.read_csv(
        "result/generated_perturbation_list/" +
        args.pert_file_con+"/"+file_name, header=0, sep='\t')
    
    if int(args.perturbation_subset_count) > 0:
        # set seed
        set_seed(42)
        con_pert = con_pert.sample(int(args.perturbation_subset_count))

    pert_list = set(con_pert[con_pert.columns[0]])

    if not os.path.exists(args.result_path+'evaluation/' +
                          args.experiment+'/img_quality_result/'):
        os.makedirs(args.result_path+'evaluation/' +
                    args.experiment+'/img_quality_result/')

    if args.per_pert:
        print(pert_list)

        result_path = create_result_file(args)
        print('Calculating image quality metrics per perturbation.')

        for pert in pert_list:
            print('Perturbation: '+pert)
            evaluate_generated_images_per_pert(
                pert,
                args.ref_img_path,
                args.gen_img_path,
                args.img_aug_count,
                args.experiment,
                model_list,
                result_path)

    return


if __name__ == "__main__":
    main()
