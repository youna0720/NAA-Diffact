"""implements a dataset object which allows to read representations from LMDB datasets in a multi-modal fashion
The dataset can sample frames for both the anticipation and early recognition tasks."""

import torch
import pdb
import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils import data
import pandas as pd
from collections import defaultdict

def read_representations(frames, env, tran=None):
    """ Reads a set of representations, given their frame names and an LMDB environment.
    Applies a transformation to the features if provided"""
    features = []
    # for each frame
    for f in frames:
        # read the current frame
        with env.begin() as e:
            dd = e.get(f.strip().encode('utf-8'))
        if dd is None:
            print(f)
            pdb.set_trace()
            continue
        # convert to numpy array
        data = np.frombuffer(dd, 'float32')
        # append to list
        features.append(data)
    # convert list to numpy array
    features=np.array(features)
    # apply transform if provided
    if tran:
        features=tran(features)
    return features

def read_maxpooled_representations(frames_list, env):
    features = []
    missing_Fidx = []
    for idx, frames in enumerate(frames_list):
        pooling_feature = []
        for f in frames:
            with env.begin() as e:
                dd = e.get(f.strip().encode('utf-8'))
            if dd is None:
                continue
            else:
                pooling_feature.append(np.frombuffer(dd, 'float32'))
        if(len(pooling_feature)==0):
            missing_Fidx.append(idx)
            features.append(np.zeros(1024))
        else:
            pooling_feature = np.array(pooling_feature)
            pooled_feature = np.max(pooling_feature, 0)
            features.append(pooled_feature.squeeze())    #need squeeze?

    #print(str(len(missing_Fidx))+"/"+str(len(frames_list)))
    for midx in missing_Fidx[::-1]:
        if midx != len(frames_list)-1:
            features[midx] = features[midx+1]
    features = np.stack(features)
    return features

def frames_to_flist(frames):
    frames_list = []
    for idx in range(len(frames)):
        if(idx!=0):
            start = frames[idx-1]+1
        else:
            if frames[idx]-(frames[idx+1]-frames[idx])>0:
                start = frames[idx]-(frames[idx+1]-frames[idx])
            else:
                start = 1
        frs = np.arange(start, int(frames[idx])+1, 1)
        frames_list.append(frs)
    return frames_list

def read_data(frames, env, tran=None, maxpool=False):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., RGB + Flow)"""

    if maxpool==True:
        if isinstance(env, list):
            l = [read_maxpooled_representations(frames, e) for e in env]
            return l
        else:
            return read_maxpooled_representations(frames, env)

    else:
        # if env is a list
        if isinstance(env, list):
            # read the representations from all environments
            l = [read_representations(frames, e, tran) for e in env]
            return l
        else:
            # otherwise, just read the representations
            return read_representations(frames, env, tran)

class SequenceDataset(data.Dataset):
    def __init__(self, path_to_lmdb, path_to_csv, num_class, label_type = 'action',
                time_step = 0.25, sequence_length = 14, fps = 30,
                img_tmpl = "frame_{:010d}.jpg", ta = 0.25, pad_idx = 2515,
                transform = None,
                challenge = False,
                past_features = True,
                action_samples = None):
        """
            Inputs:
                path_to_lmdb: path to the folder containing the LMDB dataset
                path_to_csv: path to training/validation csv
                label_type: which label to return (verb, noun, or action)
                time_step: in seconds
                sequence_length: in time steps
                fps: framerate
                img_tmpl: image template to load the features
                tranform: transformation to apply to each sample
                challenge: allows to load csvs containing only time-stamp for the challenge
                past_features: if past features should be returned
                action_samples: number of frames to be evenly sampled from each action
        """

        # read the csv file
        if challenge:
            self.annotations = pd.read_csv(path_to_csv, header=None, names=['video','start','end'])
        else:
            self.annotations = pd.read_csv(path_to_csv, header=None, names=['video','start','end','verb','noun','action'])

        self.n_class = num_class

        self.challenge=challenge
        self.path_to_lmdb = path_to_lmdb
        self.time_step = time_step
        self.past_features = past_features
        self.action_samples = action_samples
        self.fps=fps
        self.transform = transform
        self.label_type = label_type
        self.sequence_length = sequence_length
        self.img_tmpl = img_tmpl
        self.action_samples = action_samples
        self.ta = ta
        self.pad_idx = pad_idx

        # initialize some lists
        self.ids = [] # action ids
        self.discarded_ids = [] # list of ids discarded (e.g., if there were no enough frames before the beginning of the action
        self.discarded_labels = [] # list of labels discarded (e.g., if there were no enough frames before the beginning of the action
        self.past_frames = [] # names of frames sampled before each action
        self.past_pooling_frames = []
        self.action_frames = [] # names of frames sampled from each action
        self.labels = [] # labels of each action
        self.sparse_dict = defaultdict(list)    #dict for sparse label
        self.vname_list = []        #matching samples and its vname in get_item
        self.frame_num = []         #before parsing past_frames

        # populate them
        self.__populate_lists()

        # if a list to datasets has been provided, load all of them
        if isinstance(self.path_to_lmdb, list):
            self.env = [lmdb.open(l, readonly=True, lock=False) for l in self.path_to_lmdb]
        else:
            # otherwise, just load the single LMDB dataset
            self.env = lmdb.open(self.path_to_lmdb, readonly=True, lock=False)

        self.__getitem__(0)

    def __get_frames(self, frames, video):
        """ format file names using the image template """
        frames = np.array(list(map(lambda x: video+"_"+self.img_tmpl.format(x), frames)))
        return frames

    def __populate_lists(self):
        """ Samples a sequence for each action and populates the lists. """
        for _, a in tqdm(self.annotations.iterrows(), 'Populating Dataset', total = len(self.annotations)):

            # sample frames before the beginning of the action
            frames = self.__sample_frames_past(a.start)
            #get video's start, end, action label as dictionary
            self.sparse_dict[a.video].append([a.start, a.end, a.action])
            self.vname_list.append(a.video)
            self.frame_num.append(frames)

            if self.action_samples:
                # sample frames from the action
                # to sample n frames, we first sample n+1 frames with linspace, then discard the first one
                action_frames = np.linspace(a.start, a.end, self.action_samples+1, dtype=int)[1:]

            # check if there were enough frames before the beginning of the action
            if frames.min()>=1: #if the smaller frame is at least 1, the sequence is valid
                self.past_frames.append(self.__get_frames(frames, a.video))

                _flist = frames_to_flist(frames)
                flist = []
                for _f in _flist:
                    flist.append(self.__get_frames(_f, a.video))
                self.past_pooling_frames.append(flist)

                self.ids.append(a.name)
                # handle whether a list of labels is required (e.g., [verb, noun]), rather than a single action
                if isinstance(self.label_type, list):
                    if self.challenge: # if sampling for the challenge, there are no labels, just add -1
                        self.labels.append(-1)
                    else:
                        # otherwise get the required labels
                        self.labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    if self.challenge:
                        self.labels.append(-1)
                    else:
                        self.labels.append(a[self.label_type])
                if self.action_samples:
                    self.action_frames.append(self.__get_frames(action_frames, a.video))
            else:
                #if the sequence is invalid, do nothing, but add the id to the discarded_ids list
                self.discarded_ids.append(a.name)
                if isinstance(self.label_type, list):
                    if self.challenge: # if sampling for the challenge, there are no labels, just add -1
                        self.discarded_labels.append(-1)
                    else:
                        # otherwise get the required labels
                        self.discarded_labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    if self.challenge:
                        self.discarded_labels.append(-1)
                    else:
                        self.discarded_labels.append(a[self.label_type])

    def __sample_frames_past(self, point):
        """Samples frames before the beginning of the action "point" """
        # generate the relative timestamps, depending on the requested sequence_length
        # e.g., 2.  , 1.75, 1.5 , 1.25, 1.  , 0.75, 0.5 , 0.25    when sequence_length=
        # in this case "2" means, sample 2s before the beginning of the action

        end_time_stamp = point/self.fps

        time_stamps = np.arange(self.ta, self.ta+self.time_step*(self.sequence_length),self.time_step)[::-1]
        time_stamps = end_time_stamp-time_stamps

        #end_obs = end_time_stamp - self.time_step
        #time_stamps = np.linspace(1/self.fps, end_obs, self.sequence_length+1)[1:]

        # convert timestamps to frames
        # use floor to be sure to consider the last frame before the timestamp (important for anticipation!)
        # and never sample any frame after that time stamp
        frames = np.floor(time_stamps*self.fps).astype(int)

        if frames.max()>=1:
            frames[frames<1]=frames[frames>=1].min()

        return frames

    def __len__(self):
        return len(self.ids)

    def ek_embed(self, content):
        #ek55 base: n_class 2513, ek100: 3806
        emb_out = np.zeros([1, int(self.n_class)])
        for idx, i in enumerate(emb_out[0]):
            if (idx==content[0]):
                emb_out[0][idx] = 1
        return emb_out

    def slabel_embed(self, content):
        emb_out = np.zeros([1, int(self.n_class)])
        for idx, i in enumerate(emb_out[0]):
            for con in content:
                if (idx==con):
                    emb_out[0][idx] = 1
        return emb_out

    def __getitem__(self, index):
        """ sample a given sequence """
        # get past frames
        past_frames = self.past_frames[index]
        past_pooling_frames = self.past_pooling_frames[index]
        vname = self.vname_list[index]
        frame = self.frame_num[index]

        #get labels from sparse_dict if it is available
        sparse_label = []
        for fidx in frame:
            slabel=-100
            dist_cent = 99999
            for lab_list in self.sparse_dict[vname]:
                center = (lab_list[0]+lab_list[1])/2
                if lab_list[0]<=fidx:       #if fidx is bigger than "start"
                    if lab_list[1]>=fidx:   #if fidx is smaller than "end"
                        if abs(center-fidx)<dist_cent:  #if distance to center is closer then before
                            slabel = lab_list[2]        #update "action"
                            dist_cent = abs(center-fidx)
            sparse_label.append(slabel)

        item = {}
        maxpool=False
        if(maxpool==True):
            dd = read_data(past_pooling_frames, self.env, self.transform, maxpool)
        else:
            dd = read_data(past_frames, self.env, self.transform, maxpool)
        item['features'] = torch.Tensor(dd)
        label = self.labels[index]
        item['past_label'] = torch.Tensor(sparse_label)
        item['target'] = label

        return item

    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_future_label = [item['decoder_input'] for item in batch]
#
        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=self.pad_idx) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True, padding_value=self.pad_idx) #[B, S, C]
        b_future_label = torch.nn.utils.rnn.pad_sequence(b_future_label, batch_first=True, padding_value=self.pad_idx) #[B, S, C]

        batch = [b_features, b_past_label, b_future_label]


        return batch
