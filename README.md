# Is Saliency Really Captured By Sensitivity?

This repository contains the code for the paper "Is Saliency Really Captured by Sensitivity?" by Nehal Yasin, Jonathan Hare, and Antonia Marcu. In this work, we created an artificial dataset and demonstrated through validation that current gradient-based methods for feature importance, which attribute feature sensitivity to saliency, may not accurately capture true importance. Through extensive experiments, we show that popular gradient-based attribution methods, such as gradient magnitude, gradient × input, and integrated gradients, do not consistently attribute the truly important features. The image below illustrates our dataset, methodologies, followed by implementation guide and results.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c3b92bdb-c07b-4076-9450-51b9fa772a0d" width="504" />
</p>
<p align="center" style="margin-top: 10px;">
  (a) A specially crafted single discriminative pixel is inserted to allow models to learn a shortcut. 
  (b) An attribution method is used to predict the pixel most highly attributed to the model’s prediction. 
  (c) Additive importance compares the difference in accuracy between keeping only the discriminative pixel 
  and setting others to black, against keeping only the attributed pixel. 
  (d) Subtractive importance sets the discriminative and additive pixels to black respectively whilst keeping 
  the remaining pixels. The importance measures determine if the attributed pixel is more or less important than 
  the discriminative pixel. A good attribution method should have an accuracy at least as high as the accuracy 
  determined by the discriminative pixel.
</p>

## Install Dependencies

    pip install -r requirements.txt
    
## Dataset 

The `datasets.singlepixel.SinglePixelDataset` contains the dataset implementation 
described in the paper

## Training models

Make the output directory

    mkdir outputs

To train a single model:

    COL=0
    POS=0
    SEED=0
    DATASEED=0
    python single_pixel_experiments.py --log-dir=outputs --data-dir=./data --size=32 --model=resnet18_3x3 --seed="$SEED" --data-seed="$DATASEED" --position-sd=$POS --colour-sd=$COL --max-epochs=200 --checkpoint-period=10 --dataset-class SinglePixelDataset

Repeat for a range of different models (`COL` and `POS`are the s.d. for the colour 
and position of the discriminative pixel, `SEED` controls model init randomness, 
`DATASEED` controls dataset randomness (and the pixel)). Other arguments should be
self-explanatory. 

## Analysing models

For each saved model run `grad_measures.py` with the path to the saved model - e.g.:

    python grad_measures.py output/resnet18_3x3-32-seed_0-dataseed_0-pos_sd_0.0-col_sd_0.0-momentum_0.9-wd_0-SinglePixelDataset 

## Collating results

To create a single csv file from every model analysed:
    
    awk '(NR == 1) || (FNR > 1)' outputs/*/attribution_results.csv > all_results.csv

## Summarise results

Use the `attribution_analysis.ipynb` notebook to load and filter the `all_results.csv` file.

`all_results.csv contains the results for all the experiments systematically. Following table explains what each column represents.

| Header  | What it is about? |
| ------------- | ------------- |
| overall.withpix_model_accuracy  | Accuracy on the dataset with discriminative pixel added |
| overall.attribution_correct_accuracy  | Number of images the attribution method correctly predicted the discriminative pixel  |
| overall.nopix_model_accuracy |   Accuracy of model on original data (without any discriminative pixel added)   |
| overall.truebg_withblackpix_model_accuracy | Accuracy of the model on images with true/original background by replacing the attributed pixel with a black pixel  |
| The following experiments are performed only for those images where attribution methods did not correctly identified the discriminative pixel as the attributed one |
| attribution_incorrect.nopix_model_accuracy | Accuracy of the model on original data without any discriminative pixel |
| attribution_incorrect.withpix_model_accuracy | Accuracy of the model with the discriminative pixel added |
| attribution_incorrect.blackbg_withpix_model_accuracy | Accuracy of the model on a completely black background with only discriminative pixel present |
| attribution_incorrect.blackbg_withattrpix_model_accuracy | Accuracy of the model on a completely black background with only the attributed pixel present   |
| attribution_incorrect.truebg_withblackattrpix_model_accuracy | Accuracy of the model with original background but replacing the attributed pixel with a black pixel |
| attribution_incorrect.truebg_withblackpix_model_accuracy |  Accuracy of the model with original background but replacing the discriminative pixel with a black pixel  |
| attribution_incorrect.truebg_withblackattrpix_preserved_accuracy | The percentage of times the prediction doesn't change when setting the attributed pixel to black   |
| attribution_incorrect.truebg_withblackpix_preserved_accuracy | The percentage of times the prediction doesn't change when setting the discriminative pixel to black   |

