import tensorflow as tf
from random import randrange, uniform


def triplet_loss_func(y_true, y_pred, alpha=0.3):
	'''
	Used directly as loss function
		Inputs:
					y_true: True values of classification. (y_train)
					y_pred: predicted values of classification.
					alpha: Distance between positive and negative sample, arbitrarily
						   set to 0.3

		Returns:
					Computed loss

		Function:
					--Implements triplet loss using tensorflow commands
					--The following function follows an implementation of Triplet-Loss 
					  where the loss is applied to the network in the compile statement 
					  as usual.
	'''
	anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

	positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
	negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), -1)
	

	loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(loss_1, 0.0))

	return loss


def triplet_loss_fn(x, alpha=0.3):
	'''
	This is not used in given implementation.
		
	If used, used as the mode of merging.

		Inputs:
					y_true: True values of classification. (y_train)
					y_pred: predicted values of classification.
					alpha: Distance between positive and negative sample, arbitrarily
						   set to 0.3

		Returns:
					Computed loss

		Function:
					--Implements triplet loss using tensorflow commands
					--The following function follows an implementation of Triplet-Loss 
					  where the loss is applied to three separate image-embeddings, in a merge 
					  layer. 
	'''
	anchor, positive, negative = x

	positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
	negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), 1)

	loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(loss_1, 0.0), 0)

	return loss


def randindex() :
	index = []
	for i in range(randrange(0,8)) :
		index.append(randrange(0, 80))

	return index



def randlabel(nbr_labels) :
	
	frame_labels = []
	for i in range(nbr_labels) :
		index = randindex()

		if i in index :
			frame_labels.append(1.0)
		else :
			frame_labels.append(0.0)
	
	return frame_labels
	

	

def randclip(keyframe, nbr_labels) :

	clip_labels = []
	for i in range(keyframe) :
		clip_labels.append(randlabel(nbr_labels))

	return clip_labels


def randfinal(nbr_clips, keyframe, nbr_labels) :

	res = []
	for i in range(nbr_clips) :
		res.append(randclip(keyframe, nbr_labels))
	
	return res


def main() :

	nbr_clips, keyframe, nbr_labels = 20, 31, 80

	anchor_true = randfinal(nbr_clips, keyframe, nbr_labels) 
	positive_true = anchor_true
	for i in range(len(anchor_true)) :
		if i%10 == 0 :
			positive_true[i] == anchor_true[i-2]

			
	negative_true = randfinal(nbr_clips, keyframe, nbr_labels) 
	anchor_pred = randfinal(nbr_clips, keyframe, nbr_labels) 
	positive_pred = anchor_pred
	for i in range(len(anchor_pred)) :
		if i%10 == 0 :
			positive_pred[i] == anchor_pred[i-2]


	negative_pred = randfinal(nbr_clips, keyframe, nbr_labels) 
	y_true = [anchor_true, positive_true, negative_true]
	y_pred = [anchor_pred, positive_pred, negative_pred]
	
	loss = triplet_loss_func(y_true, y_pred, alpha=0.3)

	print(loss)



main()