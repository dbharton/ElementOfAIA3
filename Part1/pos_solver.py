###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids:
# Davyn Hartono - dbharton
# Wooserk Park - wp2
# (Based on skeleton code by D. Crandall)
#


import random
import math
import sys
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

# Solver function
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        # In the probability calculation, we use -log probability
        # The highest probability is the one that has the minimum value of -log probability
        join_prob = 0
        if model == "Simple":
            for t, (w, tag) in enumerate(zip(sentence, label)):
                join_prob += self.simple_prob(w,tag)
            return join_prob
        elif model == "HMM":
            prevtag = label[-1]
            for t, (w, tag) in enumerate(zip(sentence, label)):
                join_prob += self.hmm_prob(t,w,tag,prevtag)
            return join_prob
        elif model == "Complex":
            return self.complex_posterior(sentence,label)
        else:
            print("Unknown algo!")

    alpha_simple = 0.001
    alpha_hmm = 0.0001
    alpha_complex = 0.05
    # Function of P(tag|word) in simple model
    def simple_prob(self, word, tag):
        # We ignore the marginal (denominator) since the value is constant accross all classes
        # Only likelihood * prior that determines which class a word should be assigned
        # Laplace Smoothing is used in the probability calculation to avoid 0 probability due to unmatch testing set's words in train testing set
        # Alpha value is determined by doing multiple run and choosing the value that gives the highest probability
        # By trying several values, the best alpha value is 0.001
        alpha = self.alpha_simple
        # Probability of a speech of tag (tag_i)
        C_tag = self.tag_prob(tag)
        # Probability of a word given a tag (P(word|tag)) --> Emission Probability
        C_w_tag = self.emission_prob(word, tag, alpha)
        # Probability of P(tag|word)
        C_tag_w = C_w_tag + C_tag
        return C_tag_w

    # Function of P(tag|word) in HMM model
    def hmm_prob(self, timestamp, word, tag, prevtag='noun'):
        # The optimal alpha value is 0.0001 after several iteration
        alpha = self.alpha_hmm
        # Emission Probability
        C_w_tag = self.emission_prob(word, tag, alpha)
        if timestamp == 0:
            # Initial probability
            C_tag0 = self.initial_prob(tag)
            C_tag_w = C_tag0 + C_w_tag # Probability of P(tag|word) at t
        elif timestamp > 0:
            # Transition probability
            C_tag_prevtag = self.trans_prob(prevtag, tag, alpha)
            C_tag_w = C_w_tag + C_tag_prevtag # Probability of P(tag|word) at t
        return C_tag_w

    # P(tag t+1|tag t)
    def trans_prob_next(self,tag_sequence, current_tag, timestamp, alpha):
        try:
            return self.trans_prob(current_tag, tag_sequence[timestamp + 1],alpha)
        except IndexError:
            return 0

    # P(tag t+2|tag t+1,tag t)
    def trans_prob_next2_gibbs(self,tag_sequence, current_tag, timestamp, alpha):
        try:
            return self.trans_prob_gibbs(current_tag, tag_sequence[timestamp + 1], tag_sequence[timestamp + 2], alpha)
        except IndexError:
            return 0

    # P(w t+1|tag t+1,tag t)
    def emission_prob_next(self,tag_sequence, sentence, word, current_tag, alpha, timestamp):
        try:
            return self.emission_prob_gibbs(sentence[timestamp + 1], current_tag, tag_sequence[timestamp + 1], alpha)
        except IndexError:
            return 0

    # Calculating complex model posterior
    def complex_posterior(self, sentence, label):
        alpha = self.alpha_complex
        C_tot = 0
        for t, (w, tag) in enumerate(zip(sentence, label)):
            C_tagnext2_nexttag_tagt = self.trans_prob_next2_gibbs(label, tag, t, alpha)  # P(tag t+2|tag t)
            C_wtnext_tagnext_tagt = self.emission_prob_next(label,sentence, w, tag, alpha,t)  # P(w t+1|tag t+1,tag t)
            if t == 0:
                C_tag0 = self.initial_prob(tag)  # Initial probability
                C_tagnext_tagt = self.trans_prob_next(label, tag, t, alpha)  # P(tag t+1|tag t)
                C_wt_tagt = self.emission_prob(w, tag, alpha)  # P(w t|tag t)
                C_tot += C_tag0 + C_tagnext_tagt + C_wt_tagt + C_tagnext2_nexttag_tagt + C_wtnext_tagnext_tagt
            else:
                C_tot += C_tagnext2_nexttag_tagt + C_wtnext_tagnext_tagt
        return C_tot

    # Do the training!
    #
    def train(self, data):
        # Preparing training data set into dictionary.
        tr_dict = {} # Count value given tag at t
        tr_dict_gibbs = {} # Count value given tag at t and t-1
        tr_gt_dict = {} # Count value of tag at t
        tr_gt_dict_gibbs = {} # Count value of tag at t and t-1
        tr_s_dict = {} # Count value of word
        # Inital and transition probability
        prob = {}
        for (s,gt) in data:
            for index,(w,tag) in enumerate(zip(s,gt)):
                # Creating count value dictionary for combination of speech of tag & word
                key = str(w)+'_'+str(tag)
                tr_dict[key] = tr_dict.get(key,0)+1
                # Creating count value dictionary for speech of tag
                tr_gt_dict[tag] = tr_gt_dict.get(tag,0)+1
                # Creating count value dictionary for each word
                tr_s_dict[w] = tr_s_dict.get(w,0)+1
                # Counting value for gibbs sampling that consists of timestamp at t, t-1, and t-2
                if index > 0:
                    # Creating count value dictionary for for word given speech of tag at t and t-1
                    key_gibbs = str(w) + '_' + str(gt[index-1]) + '_' + str(tag)
                    tr_dict_gibbs[key_gibbs] = tr_dict_gibbs.get(key_gibbs,0)+1
                    # Creating count value dictionary for speech of tag at t-1 and t
                    key_tag_gibbs = str(gt[index-1]) + '_' + str(tag)
                    tr_gt_dict_gibbs[key_tag_gibbs] = tr_gt_dict_gibbs.get(key_tag_gibbs,0)+1
        return tr_dict, tr_gt_dict, tr_s_dict, tr_dict_gibbs, tr_gt_dict_gibbs

    # Function for initial and transition probability
    def trans_initial_prob(self, data):
        trans_prob = {}
        initial_prob = {}
        trans_prob_gibbs = {}
        tot_sentence = len(data)
        for (s, gt) in data:
            tag_pair = zip(gt, gt[1:]) # Pair of speech of tag at t-1 and t
            tag_pair2 = zip(gt, gt[1:], gt[2:]) # Pair of speech of tag at t-2, t-1 and t
            initial_prob[gt[0]] = initial_prob.get(gt[0],0)+(1/tot_sentence)
            # tag1 = tag for timestamp t-1
            # tag2 = tag for timestamp t
            for tag1,tag2 in tag_pair:
                key = tag1+'_'+tag2
                trans_prob[key] = trans_prob.get(key,0)+(1/tr_gt_dict.get(tag1))
            # tag1 = tag for timestamp t-2
            # tag2 = tag for timestamp t-1
            # tag3 = tag for timestamp t
            for tag1,tag2,tag3 in tag_pair2:
                key = tag1 + '_' + tag2 + '_' + tag3
                trans_prob_gibbs[key] = trans_prob_gibbs.get(key, 0) + (1 / tr_gt_dict_gibbs.get(tag1 + '_' + tag2))
        return trans_prob, initial_prob, trans_prob_gibbs

    # Functions for Emission Probability using Laplace Smooting to avoid 0 probability due to unmatch words between testing set and training set
    # We use C = -log probability to transform probability value into log value to avoid underflow
    def emission_prob(self, word, tag, alpha):
        count_w_tag = tr_dict.get(str(word) + '_' + str(tag), 0)
        count_tag = tr_gt_dict.get(tag)
        C_w_tag = -math.log((count_w_tag + alpha) / (count_tag + len(tr_s_dict) * alpha))
        return C_w_tag

    # Function for calculating Emission Probability on complex model (given tag at t and t-1)
    def emission_prob_gibbs(self, word, prevtag, tag, alpha):
        count_w_tag = tr_dict_gibbs.get(str(word) + '_' + str(prevtag) + '_' + str(tag), 0)
        count_tag = tr_gt_dict_gibbs.get(str(prevtag) + '_' + str(tag), sum(tr_gt_dict_gibbs.values()))
        C_w_tag = -math.log((count_w_tag + alpha) / (count_tag + len(tr_s_dict) * alpha))
        return C_w_tag

    # Function for initial probability
    def initial_prob(self,tag):
        C_tag0 = -math.log(initial_prob.get(tag))
        return C_tag0

    # Function for tag probability
    def tag_prob(self,tag):
        count_tag = tr_gt_dict.get(tag, 0)
        total_tag = sum(tr_gt_dict.values())
        C_tag = -math.log(count_tag / total_tag)
        return C_tag

    # Function for transition probability given tag at t-1
    def trans_prob(self, prevtag, tag, alpha):
        # It is possible that a combination of transition speech of tag is not accommodated in training sets
        # Example: there are no transition from DET to VERB that could be found in training sets
        # This condition will result 0 probability. If we use -log probability, then the value will be undefined.
        # Therefore, we also use Laplace Smoothing for this condition.
        C_tag_prevtag = -math.log(trans_prob.get(prevtag + '_' + tag, (alpha / (tr_gt_dict.get(prevtag) + len(tr_gt_dict) * alpha))))
        return C_tag_prevtag

    #Function for transition probability given tag at t-1 and t-2
    def trans_prob_gibbs(self, prev2tag, prevtag, tag, alpha):
        C_tag_prevtag_prev2tag = -math.log(trans_prob_gibbs.get(prev2tag + '_' + prevtag + '_' + tag
                                    , (alpha/(tr_gt_dict_gibbs.get(prev2tag+'_'+prevtag ,sum(tr_gt_dict_gibbs.values()))+len(tr_gt_dict)*alpha))))
        return C_tag_prevtag_prev2tag

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        # Label_dict is a dictionary for storing the final tag classification for all words
        label_dict = {}
        label=[]
        for w in sentence:
            # Each word we check into label_dict: if a word is already stored in the dict, then we can just call the value.
            # If not, we should calculate the probability
            if w in label_dict:
                label.append(label_dict.get(w))
            else:
                # option_label_dict is used to store all the possibilites of tag
                option_label_dict = {}
                for tag in tr_gt_dict:
                    option_label_dict[tag] = self.simple_prob(w,tag)
                # Label_dict is a dictionary for storing the final tag classification on each word
                label_dict[w] = min(option_label_dict,key=option_label_dict.get)
                label.append(label_dict[w])
        return label

    def hmm_viterbi(self, sentence):
        # Creating a matrix template for virtebi's result with row = word/timestamp, col = speech of tag
        virtebi = np.zeros((len(sentence),len(tr_gt_dict)))
        # Storing a sequence of tag that gives the maximum probability in a list of tuples
        tag_seq = []
        for t, w in enumerate(sentence):
            tag_seq_t = []
            for index_tag, tag in enumerate(tr_gt_dict):
                if t == 0:
                    virtebi[t][index_tag] = self.hmm_prob(t,w,tag)
                    tag_seq_t.append((tag,))
                else:
                    # All possibilities for transition tag
                    option = {}
                    for index_prevtag, prevtag in enumerate(tr_gt_dict):
                        option_key = tag_seq[index_prevtag]+(tag,)
                        option[option_key] = virtebi[t-1][index_prevtag] + self.hmm_prob(t,w,tag,prevtag)
                    # Updating value of virtebi matrix with max probability
                    virtebi[t][index_tag] = min(option.values())
                    # Adding new sequence of tag at timestamp t to temporary array
                    tag_seq_t.append(min(option,key=option.get))
            # Adding new sequence of tag at timestamp t to final array
            tag_seq = tag_seq_t
        min_cost_index = np.argmin(virtebi[-1])
        label = tag_seq[min_cost_index]
        return label

    def complex_mcmc(self, sentence):
        # Storing n sample for every iteration after burn-in time which is a condition where stationary distribution exists
        sample_tag = []
        # Assigning initial tag for each word to be iterated.
        # In this process, we use estimated tag by HMM model since it gives higher accuracy than simple model.
        # Hopefully, by assigning label from HMM model, it could reduce the number of iteration (reach the stationary state quicker)
        label = list(self.hmm_viterbi(sentence))
        # The chosen number of iteration and burn-in time is based on multiple run.
        # This value is selected as it is the lowest number of iteration that does not change (or change a bit) the accuracy.
        iteration = 10
        burn_time = 5
        for iter in range(0,iteration):
            for index_w, w in enumerate(sentence):
                option = {}
                for index_tag, tag in enumerate(tr_gt_dict):
                    label[index_w] = tag
                    option[tag] = self.complex_posterior(sentence,label)
                label[index_w] = min(option, key=option.get)
            if iter > burn_time:
                # Storing the iteration result as a stationary state sample to see which tag on each word that has the highest frequency
                sample_tag.append(label)
        # Choosing tag from the sample that gives the highest probability (minimum cost or minimum -log probability)
        sample_tag = np.transpose(sample_tag).tolist()
        final_tag = [max(i, key = i.count) for i in sample_tag]
        return final_tag

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

# Read function for training dataset
def read_data(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]

    return exemplars

# Returning train function
(train_file, test_file) = sys.argv[1:3]
train_data = read_data(train_file)
tr_dict, tr_gt_dict, tr_s_dict, tr_dict_gibbs, tr_gt_dict_gibbs = Solver().train(train_data)

# Returning transition and inital probability function
trans_prob, initial_prob, trans_prob_gibbs = Solver().trans_initial_prob(train_data)