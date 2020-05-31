''' This function print the main results for our method '''
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

f_eval = open("segmentation_evaluation.csv", 'w')
f_eval.write('Img, Accuracy, Dice, Jaccard, Sensitivity, Specificity'+'\n')

print('-'*30)
print('Segmentation evaluation...')
print('-'*30)

print('Img, Accuracy, Dice, Jaccard, Sensitivity, Specificity')

# Initialization
sum_acc = 0
sum_dice = 0
sum_jacc = 0
sum_sensitivity = 0
sum_specificity = 0

for i in range(1,379):

    ### path to ground truth labels
    labels_path = './csv/original_'+str(i)+'.txt'

    ### path to class predictions file
    class_predictions_path = './csv/masks_'+str(i)+'.csv'

    ### Read files
    y_true = np.loadtxt(labels_path)
    y_pred = np.loadtxt(class_predictions_path)

    # Compute TP, FP, TN, FN
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))

    #print('-'*30)
    #print('False Negatives: ', str(FN))
    #print('-'*30)
    #print('True Positives: ', str(TP))
    #print('-'*30)
    #print('False Positives: ', str(FP))
    #print('-'*30)
    #print('True Negatives: ', str(TN))
    #print('-'*30)

    # Compute Accuracy score.
    acc = metrics.accuracy_score(y_true, y_pred)
    sum_acc = sum_acc + acc

    # Compute Dice score.
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dice = 2. * intersection.sum() / (im1.sum() + im2.sum())
    sum_dice = sum_dice + dice

    # Compute Jaccard similarity coefficient score
    jacc = metrics.jaccard_similarity_score(y_true, y_pred)
    sum_jacc = sum_jacc + jacc

    # Compute the Sensitivity (recall)
    sensitivity = TP/float(TP + FN)
    sum_sensitivity = sum_sensitivity + sensitivity

    # Compute Specificity score.
    specificity = TN/float(TN+FP)
    sum_specificity = sum_specificity + specificity

    f_eval.write(str(i)+', ' +str(acc)+', '+str(dice)+', '+str(jacc)+', '+str(sensitivity) + ', '+str(specificity) + '\n')
    print(str(i)+', ' +str(acc)+', '+str(dice)+', '+str(jacc)+', '+str(sensitivity) + ', '+str(specificity) + '\n')

avg_acc = sum_acc / 379
avg_dice = sum_dice / 379
avg_jacc = sum_jacc / 379
avg_sensitivity = sum_sensitivity / 379
avg_specificity = sum_specificity / 379

f_eval.write('\n'+'\n'+'Average evaluation: '+str(avg_acc)+', '+str(avg_dice)+', '+str(avg_jacc)+', '+str(avg_sensitivity) + ', '+str(avg_specificity) + '\n')
f_eval.close()
