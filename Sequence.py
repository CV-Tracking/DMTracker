import os
import numpy as np
class Sequence_t(object):

    def __init__(self, name: str, dataset='/ssd3/lz/TMM2022/dataset/alldata/'):
        self._name = name
        self._dataset = dataset
        self._path = self._dataset + self._name
        self._numframe = self.num_frame

    @property
    def num_frame(self):
        try:
            seq_list = os.listdir(os.path.join(self._path, 'color'))
        except:
            print('error')
        return len(seq_list)
    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._name

    @property
    def dataset(self):
        return self._dataset

    @property
    def gt(self):
        gtfile = os.path.join(self._path, 'groundtruth.txt')
        with open(gtfile, 'r') as f:
            value = np.loadtxt(f, delimiter=',')
        
        return value 
    @property
    def invisible(self):
        full_occlusion = os.path.join(self._path, 'full-occlusion.tag')
        # out_view = os.path.join(self._path, 'out-of-frame.tag')
        out_view = os.path.join(self._path, 'full-occlusion.tag')
        if not os.path.exists(full_occlusion):
            value = np.array([0 for i in range(self._numframe)])
        else:
            with open(full_occlusion, 'r') as f:
                value_fullocc = np.loadtxt(f)
            with open(out_view, 'r') as f:
                value_outview = np.loadtxt(f)
            value = np.array([i + j for i, j in zip(value_fullocc, value_outview)])
            value[value > 1] = 1
        
        return value

    @property
    def num_inivisible(self):
        full_occlusion = os.path.join(self._path, 'full-occlusion.tag')
        # out_view = os.path.join(self._path, 'out-of-frame.tag')
        out_view = os.path.join(self._path, 'full-occlusion.tag')
        if not os.path.exists(full_occlusion):
            value = np.array([0 for i in range(self._numframe)])
        else:
            with open(full_occlusion, 'r') as f:
                value_fullocc = np.loadtxt(f)
            with open(out_view, 'r') as f:
                value_outview = np.loadtxt(f)
            value = np.array([i + j for i, j in zip(value_fullocc, value_outview)])
            value[value > 1] = 1
        if 1 in value:
            flag = True
            return flag, self._numframe
        else:
            flag = False
            
            return flag, self._numframe 