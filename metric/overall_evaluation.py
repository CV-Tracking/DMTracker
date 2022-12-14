import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging

log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/overall.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')

def compute_tpr_curves(trajectory: Tracking, sequence: Sequence_t, all_prre: PrRe, reverse=False):
    
    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.prebox_conf(sequence.name)
    gt = sequence.gt 

    if reverse:
        if trackers.name == "CSR_RGBD++":
            prebbox = prebbox[:-1]
            confidence = confidence[:-1]
        overlaps = np.concatenate((np.array([estimateIOU(prebbox[i], gt[i] ) for i in range(len(prebbox))]),[1]))
        overlaps[np.isnan(overlaps)]=0
        confidence = np.concatenate((np.array(confidence), [1]))
        
    else:     
        # firstframe in each sequence
        overlaps = np.concatenate(([1], np.array([estimateIOU(prebbox[i], gt[i+1] ) for i in range(len(prebbox))])))
        overlaps[np.isnan(overlaps)]=0
        confidence = np.concatenate(([1],np.array(confidence)))


    #n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])
    # sequence.invisible (full-occlusion tag) if invisible= 1 full-occlusion invisible
    visible = np.array(sequence.invisible) < 1
    visible = visible + 0
    try:
        assert len(overlaps) == len(visible) == len(confidence)
    except:
        print("assert not equal ", sequence.name)    
    all_prre.add_list_iou(overlaps)
    all_prre.add_visible(visible)
    all_prre.add_confidence(confidence) 


seq_list = os.listdir('/ssd3/lz/MM2022/dataset/depthtrack')
seq_list.remove('list.txt')

# seq_list = ["flag_indoor"]
# all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
# all_trackers = [Tracking(tracker) for tracker in ['DAL']]
# all_trackers = [Tracking(tracker, path="/ssd2/lz/TMM2022/reverse_result", reverse=True) for tracker in ["DAL", "DeT", "TSDM","iiau_rgbd", "CSR_RGBD++", "DSKCF_shape"]]
all_trackers = [Tracking(tracker, path="/data1/gaoshang/vot_rgbd2020/results", reverse=True) for tracker in ["DeT_DiMP50_Max"]]
all_sequence = [Sequence_t(seq) for seq in seq_list]
plotfile =  open('overall.txt', 'w')
for i, trackers in enumerate(all_trackers):
    print(trackers.name)
    for sequence in all_sequence:
        # if not sequence.name == 'pot_indoor':
            # continue
        if sequence.name in trackers._seqlist:
            compute_tpr_curves(trackers, sequence, trackers._prre, reverse=True)
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            trackers.lack(sequence.name)
            continue
    pr_list, re_list = trackers._prre.value
    pr,re,fscore = trackers._prre.fscore
    print('Trackers: {}  Seq_num: {} frame_num: {}  pr: {}  re: {}  fscore: {}'.format(trackers.name, trackers._numseq, trackers._prre.count, pr, re, fscore))
    logging.info('Trackers: {}  Seq_num: {} frame_num: {}  pr: {}  re: {}  fscore: {}'.format(trackers.name, trackers._numseq, trackers._prre.count, pr, re, fscore))
    plotfile.writelines(trackers.name)
    plotfile.writelines('\n')
    
    plotfile.writelines('Pr_list: ')
    plotfile.writelines(str(pr_list))
    plotfile.writelines('\n')
    
    plotfile.writelines('Re_list: ')
    plotfile.writelines(str(re_list))
    plotfile.writelines('\n')

plotfile.close()