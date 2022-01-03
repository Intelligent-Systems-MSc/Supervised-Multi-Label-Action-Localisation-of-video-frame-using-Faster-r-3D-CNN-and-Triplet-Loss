import tensorflow as tf
from PIL import Image
import numpy as np
import os
from Parser2 import Parser_csv, Parser_pbtxt




class Dataset() :

    def __init__(self, video_dir, video_type, height=128, width=128, keyframes=31) :
        self.frame_dir = video_dir+'/frames_'+ video_type
        self.height = height
        self.width = width
        self.keyframes = keyframes
        self.size = os.stat(self.frame_dir).st_size
        self.frames = os.listdir(self.frame_dir)
        self.dataset = []
        self.labelset = []
        self.nbr_labels = 80


    def get_clip(self, keyframes, frames) :

        clip = []
        for i in range(keyframes) :

            if len(frames) < self.keyframes :
                break

            if '.jpg' in frames[i] :
                im = Image.open(self.frame_dir+'/'+frames[i])
                im = im.resize((self.height,self.width))
                
                clip.append(tf.convert_to_tensor(im))
                #clip.append(im)

        return clip




    def str_browse(self, str) :
        
        index = 0
        for i in reversed(range(-len(str), 0)) :
            if str[i] == '_' :
                index = i
                
                break
                
        return str[:index]



    def one_to_one(self, vec_labels) :

        res = [np.zeros(self.nbr_labels)]*self.keyframes
        for t in range(self.keyframes) :
            for i in range(len(res[t])) :
                if i in vec_labels[t] :
                    res[t][i] = 1

        return res



    def get_clip_labels(self, keyframes, frames, t, P) :

        clip_labels = []
        for i in range(keyframes) :

            if len(frames) < self.keyframes :
                break

            if '.jpg' in frames[i] :
                
                try :
                    label = [int(val) for val in P.label[self.str_browse(self.frames[t])][t+i]]
                except KeyError :
                    label = [0]

                clip_labels.append(label)
                #clip_labels.append(label)

        vec_one = self.one_to_one(clip_labels)

        #vec_one = tf.ragged.constant(clip_labels)
        #vec_one = tf.convert_to_tensor(clip_labels)
        #vec_one = clip_labels


        return vec_one



    def get_dataset(self) :

        data = []

        for t in range(0, int(self.size/50), self.keyframes) :
            
            data.append(self.get_clip(self.keyframes, self.frames[t:self.keyframes+t]))

            print('\t', t*100*50/self.size, end='\r')

        self.dataset = tf.convert_to_tensor(data)
        #self.dataset = data

        return self.dataset

    
    def get_labels(self, P) :

        label = []

        for t in range(0, int(self.size/50), self.keyframes) :

            label.append(self.get_clip_labels(self.keyframes, self.frames[t:self.keyframes+t], t, P))

            print('\t', t*100*50/self.size, end='\r')

        #self.labelset = tf.ragged.constant(label)
        self.labelset = tf.convert_to_tensor(label)
        #self.labelset = label

        return self.labelset




if __name__ == '__main__':

    files = {
            'train' : 'ava_train_v2.2.csv',
            'test' : 'ava_test_excluded_timestamps_v2.2.csv',
            'val' : 'ava_val_v2.2.csv'
            }

    dir = {
          'train' : 'videos_train',
          'test' : 'videos_test',
          'val' : 'videos_val'
          }

    train_dir = dir['train']
    test_dir = dir['test']

    P = Parser_csv(files['train'])

    P.parser()


    D_train_anchor = Dataset(train_dir, 'anchor')
    D_train_positive = Dataset(train_dir, 'positive')
    D_train_negative = Dataset(train_dir, 'negative')
    #D_test = Dataset(test_dir)

    #D_train_anchor.get_dataset()
    #D_train_positive.get_dataset()
    #D_train_negative.get_dataset()
    #D_test.get_dataset()

    #D_train_anchor.get_labels(P)
    #D_train_positive.get_labels(P)
    #D_train_negative.get_labels(P)

    x_train = tf.convert_to_tensor([D_train_anchor.get_dataset(), D_train_positive.get_dataset(), D_train_negative.get_dataset()])
    y_train = tf.convert_to_tensor([D_train_anchor.get_labels(P), D_train_positive.get_labels(P), D_train_negative.get_labels(P)])

    #print(D_train_anchor.dataset)
    #print(D_train_anchor.labelset)

    #print(D_train.dataset)
    #print(D_test.dataset)

    print(x_train)
    print(y_train)