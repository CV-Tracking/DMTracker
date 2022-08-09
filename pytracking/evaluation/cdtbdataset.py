import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import re
class CDTBDDataset(BaseDataset):
    """
    CDTB, RGB dataset, Depth dataset, Colormap dataset, RGB+depth
    """
    def __init__(self, dtype='colormap'):
        super().__init__()
        self.base_path = self.env_settings.cdtb_path
        self.sequence_list = self._get_sequence_list()
        self.dtype = dtype

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        start_frame = 1

        if self.dtype == 'color':
            ext = 'jpg'
        elif self.dtype == 'rgbd':
            ext = ['jpg', 'png'] # Song not implemented yet
        else:
            ext = 'png'
        
        # anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        ground_truth_rect = ground_truth_rect[np.newaxis, :]
        # end_frame = ground_truth_rect.shape[0]
        end_frame = len(os.listdir('{}/{}/rgb'.format(self.base_path, sequence_name)))


        if self.dtype in ['colormap', 'normalized_depth', 'raw_depth', 'centered_colormap', 'centered_normalized_depth', 'centered_raw_depth']:
            group = 'depth'
        elif self.dtype == 'color':
            group = self.dtype
        else:
            group = self.dtype

        if self.dtype in ['rgb3d', 'rgbcolormap']:
            # if os.path.exists(os.path.join())
            # frames = [{'color': '{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,sequence_path=sequence_path, frame=frame_num, nz=nz),
            #            'depth': '{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,sequence_path=sequence_path, frame=frame_num, nz=nz)
            #            }for frame_num in range(start_frame, end_frame+1)]
            
            rgb = os.listdir('{}/{}/rgb'.format(self.base_path, sequence_name))
            rgb.sort(key = lambda i:int(re.match(r'(\d+)',i.split('-')[-1].split('.')[0]).group()))
            
            depth = os.listdir('{}/{}/depth'.format(self.base_path, sequence_name))
            depth.sort(key = lambda i:int(re.match(r'(\d+)',i.split('-')[-1].split('.')[0]).group()))
            
            frames = [{'color':'{base_path}/{sequence_path}/rgb/{frame_num}'.format(base_path=self.base_path,sequence_path=sequence_path, frame_num=rgb[i]),
                       'depth':'{base_path}/{sequence_path}/depth/{frame_num}'.format(base_path=self.base_path,sequence_path=sequence_path, frame_num=depth[i])
                        } for i in range(len(rgb))]

        else:
            frames = ['{base_path}/{sequence_path}/{group}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                      sequence_path=sequence_path, group=group, frame=frame_num, nz=nz, ext=ext)
                      for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        return Sequence(sequence_name, frames, 'cdtb', ground_truth_rect, dtype=self.dtype)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
#         sequence_list = [
# 'adapter01_indoor_1',
# 'backpack_blue_1',
# 'backpack_indoor_1',
# 'backpack_robotarm_lab_occ_1',
# 'backpack_room_noocc_1_1',
# 'bag01_indoor_1',
# 'bag01_indoor_2',
# 'bag01_indoor_4',
# 'bag02_indoor_1',
# 'bag02_indoor_2',
# 'bag_outside_2',
# 'bag_outside_3',
# 'ball06_indoor_1',
# 'ball06_indoor_2',
# 'ball11_wild_1',
# 'ball11_wild_2',
# 'ball11_wild_5',
# 'ball20_indoor_1',
# 'ball20_indoor_4',
# 'bandlight_indoor_1',
# 'bicycle2_outside_1',
# 'bottle_box_5',
# 'box1_outside_1',
# 'box_darkroom_noocc_10_1',
# 'box_darkroom_noocc_1_1',
# 'box_darkroom_noocc_2_1',
# 'box_darkroom_noocc_3_1',
# 'box_darkroom_noocc_4_1',
# 'box_darkroom_noocc_5_1',
# 'box_darkroom_noocc_6_1',
# 'box_darkroom_noocc_7_1',
# 'boxes_backpack_room_occ_1_2',
# 'boxes_humans_room_occ_1_1',
# 'boxes_humans_room_occ_1_4',
# 'boxes_humans_room_occ_1_5',
# 'boxes_room_occ_1_1',
# 'boxes_room_occ_1_2',
# 'box_humans_room_occ_1_1',
# 'box_room_noocc_3_1',
# 'box_room_noocc_4_1',
# 'box_room_noocc_7_1',
# 'box_room_occ_2_1',
# 'box_room_occ_2_2',
# 'cartman_1',
# 'cartman_robotarm_lab_noocc_1',
# 'case_1',
# 'cat01_indoor_1',
# 'cat01_indoor_3',
# 'colacan03_indoor_1',
# 'colacan03_indoor_3',
# 'colacan03_indoor_7',
# 'container_room_noocc_1_1',
# 'cube02_indoor_1',
# 'cube02_indoor_2',
# 'cube05_indoor_1',
# 'cube05_indoor_2',
# 'cube05_indoor_3',
# 'cube05_indoor_4',
# 'cube05_indoor_5',
# 'cube05_indoor_6',
# 'cup01_indoor_1',
# 'cup02_indoor_1',
# 'cup04_indoor_1',
# 'developmentboard_indoor_4',
# 'dog_outside_1',
# 'duck03_wild_1',
# 'duck03_wild_2',
# 'dumbbells01_indoor_1',
# 'dumbbells01_indoor_2',
# 'earphone01_indoor_1',
# 'file01_indoor_1',
# 'flag_indoor_1',
# 'glass01_indoor_1',
# 'glass01_indoor_2',
# 'human02_indoor_1',
# 'human02_indoor_2',
# 'human02_indoor_3',
# 'humans_corridor_occ_1_1',
# 'humans_shirts_room_occ_1_A_1',
# 'humans_shirts_room_occ_1_A_2',
# 'humans_shirts_room_occ_1_B_1',
# 'humans_shirts_room_occ_1_B_2',
# 'jug_1',
# 'lamp02_indoor_1',
# 'mobilephone03_indoor_2',
# 'mug_ankara_1',
# 'mug_gs_1',
# 'notebook01_indoor_1',
# 'paperpunch_3',
# 'person_outside_1',
# 'pigeon01_wild_1',
# 'pigeon02_wild_1',
# 'pigeon04_wild_1',
# 'pot_indoor_1',
# 'pot_indoor_2',
# 'pot_indoor_3',
# 'robot_corridor_occ_1_1',
# 'robot_human_corridor_noocc_1_B_1',
# 'robot_lab_occ_1',
# 'roller_indoor_1',
# 'roller_indoor_2',
# 'roller_indoor_4',
# 'shoes02_indoor_1',
# 'shoes02_indoor_2',
# 'squirrel_wild_1',
# 'squirrel_wild_2',
# 'squirrel_wild_3',
# 'stick_indoor_1',
# 'teapot_1',
# 'toiletpaper01_indoor_1',
# 'toiletpaper01_indoor_2',
# 'toy02_indoor_1',
# 'toy09_indoor_1',
# 'toy_office_noocc_1_1',
# 'trashcan_room_occ_1_2',
# 'trashcans_room_occ_1_A_1',
# 'trashcans_room_occ_1_B_1',
# 'trashcans_room_occ_1_B_3',
# 'trendNetBag_outside_1',
# 'trendNet_outside_1',
# 'trendNet_outside_2',
# 'trophy_outside_2',
# 'two_mugs_1',
# 'two_tennis_balls_1',
# 'two_tennis_balls_3',
# 'XMG_outside_2',
# 'yogurt_indoor_1',
#         ]
        # sequence_list= [
        #                 'backpack_blue',
        #                 'backpack_robotarm_lab_occ',
        #                 'backpack_room_noocc_1',
        #                 'bag_outside',
        #                 'bicycle2_outside',
        #                 'bicycle_outside',
        #                 'bottle_box',
        #                 'bottle_room_noocc_1',
        #                 'bottle_room_occ_1',
        #                 'box1_outside',
        #                 'box_darkroom_noocc_1',
        #                 'box_darkroom_noocc_10',
        #                 'box_darkroom_noocc_2',
        #                 'box_darkroom_noocc_3',
        #                 'box_darkroom_noocc_4',
        #                 'box_darkroom_noocc_5',
        #                 'box_darkroom_noocc_6',
        #                 'box_darkroom_noocc_7',
        #                 'box_darkroom_noocc_8',
        #                 'box_darkroom_noocc_9',
        #                 'box_humans_room_occ_1',
        #                 'box_room_noocc_1',
        #                 'box_room_noocc_2',
        #                 'box_room_noocc_3',
        #                 'box_room_noocc_4',
        #                 'box_room_noocc_5',
        #                 'box_room_noocc_6',
        #                 'box_room_noocc_7',
        #                 'box_room_noocc_8',
        #                 'box_room_noocc_9',
        #                 'box_room_occ_1',
        #                 'box_room_occ_2',
        #                 'boxes_backpack_room_occ_1',
        #                 'boxes_humans_room_occ_1',
        #                 'boxes_office_occ_1',
        #                 'boxes_room_occ_1',
        #                 'cart_room_occ_1',
        #                 'cartman',
        #                 'cartman_robotarm_lab_noocc',
        #                 'case',
        #                 'container_room_noocc_1',
        #                 'dog_outside',
        #                 'human_entry_occ_1',
        #                 'human_entry_occ_2',
        #                 'humans_corridor_occ_1',
        #                 'humans_corridor_occ_2_A',
        #                 'humans_corridor_occ_2_B',
        #                 'humans_longcorridor_staricase_occ_1',
        #                 'humans_shirts_room_occ_1_A',
        #                 'humans_shirts_room_occ_1_B',
        #                 'jug',
        #                 'mug_ankara',
        #                 'mug_gs',
        #                 'paperpunch',
        #                 'person_outside',
        #                 'robot_corridor_noocc_1',
        #                 'robot_corridor_occ_1',
        #                 'robot_human_corridor_noocc_1_A',
        #                 'robot_human_corridor_noocc_1_B',
        #                 'robot_human_corridor_noocc_2',
        #                 'robot_human_corridor_noocc_3_A',
        #                 'robot_human_corridor_noocc_3_B',
        #                 'robot_lab_occ',
        #                 'teapot',
        #                 'tennis_ball',
        #                 'thermos_office_noocc_1',
        #                 'thermos_office_occ_1',
        #                 'toy_office_noocc_1',
        #                 'toy_office_occ_1',
        #                 'trashcan_room_occ_1',
        #                 'trashcans_room_occ_1_A',
        #                 'trashcans_room_occ_1_B',
        #                 'trendNetBag_outside',
        #                 'trendNet_outside',
        #                 'trophy_outside',
        #                 'trophy_room_noocc_1',
        #                 'trophy_room_occ_1',
        #                 'two_mugs',
        #                 'two_tennis_balls',
        #                 'XMG_outside'
        #                 ]
#         sequence_list = [
#             'athlete_move',
# 'athlete_static',
# 'backpack_move',
# 'backpack_static',
# 'bag_move',
# 'bag_static',
# 'bin_move',
# 'bin_static',
# 'blanket_move',
# 'blanket_static',
# 'body_move',
# 'body_static',
# 'book_move',
# 'book_static',
# 'cap_move',
# 'cap_static',
# 'doll_move',
# 'doll_static',
# 'face_move',
# 'face_static',
# 'funnel_move',
# 'funnel_static',
# 'gloves_move',
# 'gloves_static',
# 'scarf_move',
# 'scarf_static',
# 'shoe_move',
# 'shoe_static',
# 'toytank_move',
# 'toytank_static',
# 'trolley_move',
# 'trolley_static',
# 'tube_move',
# 'tube_static',
# 'umbrella_move',
# 'umbrella_static',
#         ]
        sequence_list = []
        for sequence in os.listdir('/ssd2/gaoshang/dataset/EvaluationSet'):
        # for sequence in os.listdir('/data1/gaoshang/workspace_votRGBD2019/sequences'):
            sequence_list.append(sequence)
        
        print(len(sequence_list))

        return sequence_list
