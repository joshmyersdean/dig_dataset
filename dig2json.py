import json
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

with open('cocolvis_cp/val2017/hannotation.pickle', 'rb') as f:
    data = pickle.load(f)
    
    
with open('cocolvis_cp/val2017/val_imids.pkl', 'rb') as f:
    val_ids = pickle.load(f)
    
gts = os.listdir('cocolvis_cp/val2017/gts_og')
im_ids = [i[:-4] for i in gts]

im_ids = [i for i in im_ids if i in val_ids]

p_annots = sorted(os.listdir('cocolvis_cp/val2017/new_fl_syn_annotations'))
p_gts = sorted(os.listdir('cocolvis_cp/val2017/new_fl_syn_gts'))
priors = sorted(os.listdir('cocolvis_cp/val2017/new_fl_syn_init_masks'))
prior_dir = 'cocolvis_cp/val2017/new_fl_syn_init_masks/'

pa_dir = 'cocolvis_cp/val2017/new_fl_syn_annotations'
pg_dir = 'cocolvis_cp/val2017/new_fl_syn_gts'

nop_annots = sorted(os.listdir('cocolvis_cp/val2017/syn_annotations'))
nop_gts = sorted(os.listdir('cocolvis_cp/val2017/syn_gts'))
assert len(nop_annots) == len(nop_gts)
nopa_dir = 'cocolvis_cp/val2017/syn_annotations'
nopg_dir = 'cocolvis_cp/val2017/syn_gts'

def get_corr_id(fpath):
    if len(fpath.split('_')) == 5:
        return int(fpath.split('_')[2])
    return -1

def get_obj_id(fpath):
    return int(fpath.split('_')[1])

def has_prior(fpath):
    prior = fpath.split('_')[-1][:-4]
    if prior == 'noprior':
        return False
    return True

def get_mode(fpath):
    mode = fpath.split('_')[-1][:-4]
    if mode == 'sub':
        return 0
    return 1

def create_blank_row():
    row = {
            'img_id': None,
            'size': None,
            'obj_id': None,
            'syn_gt': None,
            'syn_gt_void': None,
            'syn_annotations': None,
            'has_prior': None,
            'prior': None,
            'corr_id': None,
            'gesture': None,
            'mode': None,
        }
    return row

def get_unique_priors(plist):
    priors = []
    for p in plist:
        obj = p.split('_')[1]
        check = [i for i in priors if obj == i.split('_')[1]]
        if len(check):
            continue
        priors.append(p)
    return priors

def process_prior(id_, p_annots, p_gts, priors, gesture_dict, pa_dir, pg_dir, prior_dir):
    tmp_annots_names = sorted([i for i in p_annots if id_ in i and 'mul' not in i])
    tmp_gts_names = sorted([i for i in p_gts if id_ in i and 'mul' not in i])
    plist = [i for i in priors if id_ in i]
    new_p = get_unique_priors(plist)
    prior = [cv2.imread(prior_dir + "/" + i, cv2.IMREAD_GRAYSCALE) for i in new_p]
    prior = [np.where(i > 0) for i in prior]
    priors = {}
    for i in range(len(new_p)):
        obj_id = int(new_p[i].split('_')[1])
        priors[obj_id] = prior[i]
    
    row = create_blank_row()
    row['img_id'] = id_
    _obj_ids = list(map(lambda x: get_obj_id(x), tmp_annots_names))
    row['obj_id'] = _obj_ids
    
    tmp_annots = [cv2.imread(f"{pa_dir}/{i}")[:,:,0] for i in tmp_annots_names]
    tmp_gts = [cv2.imread(f"{pg_dir}/{i}")[:,:,0] for i in tmp_gts_names]
    row['size'] = tmp_annots[0].shape
    _valids = [list(np.where(i == 1)) for i in tmp_gts]
    _voids = [list(np.where(i == 255)) for i in tmp_gts]
    row['syn_gt'] = _valids
    row['syn_gt_void'] = _voids
    
    _annots = [list(np.where(i == 255)) for i in tmp_annots]
    row['syn_annotations'] = _annots
    
    _gests = [i.split('_')[-2] for i in tmp_annots_names]
    _gests = [gesture_dict[i] for i in _gests]
    _modes = [i.split('_')[-1][:-4] for i in tmp_annots_names]
    row['gesture'] = _gests
    row['mode'] = _modes
    _corr_ids = list(map(lambda x: get_corr_id(x), tmp_annots_names))
    row['corr_id'] = _corr_ids 
    _priors = list(map(lambda x: has_prior(x), tmp_annots_names))
    row['has_prior'] = _priors
    row['prior'] = priors #[[-1] for i in _priors]
    
    _modes = list(map(lambda x: get_mode(x), tmp_annots_names))
    row['mode'] = _modes
    return row

gesture_dict = {
    'click': 0,
    'stroke': 1,
    'circle': 2,
    'boundary': 3,
    'box': 4
}

df = pd.DataFrame(columns=['img_id', 'size','obj_id', 'syn_gt', 'syn_gt_void', 'syn_annotations', 'has_prior',
                           'prior', 'corr_id', 'gesture', 'mode'])
for id_ in tqdm(im_ids):
    row = process_prior(id_, p_annots, p_gts, priors, gesture_dict, pa_dir, pg_dir, prior_dir)
    df = df.append(row, ignore_index = True)
df['img_id'] = df['img_id'].astype(str)

df.to_json("test.json", orient='records')