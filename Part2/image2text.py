#!/usr/bin/python
#
# Perform optical character recognition, usage:
#
#     python3 ./image2text.py train-image-file.png train-text.txt test-1-0.png
#     python3 ./image2text.py train-image-file.png train-text.txt test-3-0.png
#     python3 ./image2text.py train-image-file.png train-text.txt test-15-0.png
#     python3 ./image2text.py train-image-file.png train-text.txt test-17-0.png
#     python3 ./image2text.py train-image-file.png train-text.txt test-13-0.png
# 
# Authors:
#   Davyn Hartono - dbharton
#   Wooserk Park - wp2
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import math
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
total_pixels = CHARACTER_WIDTH*CHARACTER_HEIGHT

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

print('test letter size:',(len(test_letters[0]),len(test_letters[0][0])))
print('train letter size:',(len(train_letters['C']),len(train_letters['C'][0])))
print('\n')

# Load training text file from part 1 for calculating probability of a letter & transition probability
# As the prediction is case-sensitive, we didn't use lower() during reading training document
# We use text file from Assignment 2 part 2, named 'deceptive.train.txt'

def load_text(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")
    return objects

tr_text = load_text('deceptive.train.txt')

# Calculating Initial probability from text document
def prob(text):
    i_count = {}
    i_prob = {}
    letter_c = {}
    letter_prob = {}
    trans_prob = {}
    for sentence in text:
        # Calculating initial probability
        #first_w = sentence[0]
        #i_prob[first_w]=i_prob.get(first_w,0) + 1/len(text)
        # Counting letter & character
        for letter in sentence:
            letter_c[letter] = letter_c.get(letter,0) + 1
        for word in sentence.split():
            first_letter = word[0]
            i_count[first_letter] = i_count.get(first_letter, 0) + 1
    for i in i_count:
        i_prob[i] = i_count.get(i)/sum(i_count.values())
    # Calculating transition probability
    for sentence2 in text:
        for index,letter in enumerate(sentence2):
            if index > 0:
                key = sentence2[index-1]+letter
                trans_prob[key] = trans_prob.get(key,0) + 1/letter_c.get(letter)
    # Calculating probability of each character
    for j in letter_c:
        letter_prob[j] = letter_c.get(j)/sum(letter_c.values())
    return i_prob,i_count,letter_c,letter_prob, trans_prob

# Calling all parameter that is needed for the main code
i_prob,i_count,letter_c,letter_prob, trans_prob2 = prob(tr_text)

#######---------------------Emmision probability---------------------
def pixel_prob(train_letters_dict,ts_letter):
    alpha = 1
    dropout = 0.26
    option = {}
    for tr_letter in train_letters:
        white = 0
        black = 0
        for ix_row, row in enumerate(train_letters.get(tr_letter)):
            for ix_pix, pix in enumerate(row):
                if pix == ts_letter[ix_row][ix_pix]:
                    if ts_letter[ix_row][ix_pix] == '*':
                        black += 1
                    else:
                        white += 1
                    count = black + white * dropout
        tot_prob = -math.log((count + alpha) / (total_pixels + 2 * alpha))
        option[tr_letter] = tot_prob
    return option

#######---------------------Simplified---------------------
def simplified(train_letters, test_letters):
    # dropout value is a weight for ' ' character. Normally, if the value of a character in train_letters is the same as test_letters (same index),
    #   the counter will be added by 1. However, if that value is ' ', then the counter will be added by dropout value.
    # This value is selected based on several run where the optimal value is 0.26.
    ts_label = []
    ts_prob = []
    for ts_letter in test_letters:
        option = pixel_prob(train_letters,ts_letter)
        ts_label.append(min(option, key=option.get))
        ts_prob.append(min(option.values()))
    return ts_label

#######---------------------HMM model---------------------
def hmm(train_letters,test_doc):
    # Similar to simplified model, dropout value is a weight. It determines the contribution of transition and initial probability.
    # If the weight is equal to 1, the result will be false as both transition & initial probability determines the prediction,
    #   whereas it's supposed to be the image itself that contributes the most. Transition & initial probability are only as additional parameter
    #   to refine the prediction.
    # The optimal value is 0.009
    dropout=0.009
    alpha = 1
    # Creating a matrix template for virtebi's result with row = word/timestamp, col = speech of tag
    virtebi = np.zeros((len(test_doc), len(train_letters)))
    # Storing sequences of letter/character that gives the maximum probability
    letter_seq = []
    for t, ts_letter in enumerate(test_doc):
        letter_seq_temp = []
        option = pixel_prob(train_letters,ts_letter)
        for ix_opt, opt in enumerate(option):
            if t == 0:
                virtebi[t][ix_opt] = option.get(opt) + \
                                        -math.log(i_prob.get(opt, alpha/(sum(i_count.values()) + alpha*len(train_letters)) ) )*dropout
                letter_seq_temp.append(str(opt))
            else:
                # Storing combination of letter/character
                comb={}
                for ix_prev_opt, prev_opt in enumerate(option):
                    key = prev_opt+opt
                    word = letter_seq[ix_prev_opt]+opt
                    comb[word] = virtebi[t-1][ix_prev_opt] + option.get(opt)+ \
                                -math.log(trans_prob2.get(key, alpha/(letter_c.get(prev_opt,sum(letter_c.values())) + alpha*len(trans_prob2)) ) )*dropout
                virtebi[t][ix_opt] = min(comb.values())
                letter_seq_temp.append( str(min(comb,key = comb.get)) )
        letter_seq = letter_seq_temp
    min_cost_index = np.argmin(virtebi[-1])
    label = letter_seq[min_cost_index]
    return label

#----------------------- Result --------------------------
ts_label_simpl = simplified(train_letters,test_letters)
ts_label_hmm = hmm(train_letters,test_letters)

# The final output of predicted text
print("--------------- Prediction --------------")
print("Simple: " + ''.join(ts_label_simpl))
print("   HMM: " + ''.join(ts_label_hmm))


