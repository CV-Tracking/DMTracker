### Architecture
![The network of DMTracker](./imgs/structure3.png)

### Install
```
bash install.sh path-to-anaconda DMT
```

### Train
Using the default DiMP50 or ATOM pretrained checkpoints can reduce the training time.

For example, move the default dimp50.pth into the checkpoints folder and rename as DiMPNet_Det_EP0050.pth.tar

```
python run_training.py dimp DMT_DiMP50

```

### Test
```
python run_tracker.py dimp DMT_DiMP50 --dataset_name depthtrack 

```



### DepthTrack Test set (50 Sequences) 
[Download](https://doi.org/10.5281/zenodo.5792146)

### DepthTrack Training set (152 Sequences)
[Download (100 seqs)](https://doi.org/10.5281/zenodo.5794115),  [Download (52 seqs)](https://doi.org/10.5281/zenodo.5837926)

All videoes are 640x360, except 4 sequences in 640x320: painting_indoor_320, pine02_wild_320, toy07_indoor_320, hat02_indoor_320
```
@InProceedings{yan2021det,
    author    = {Yan, Song and Yang, Jinyu and Kapyla, Jani and Zheng, Feng and Leonardis, Ales and Kamarainen, Joni-Kristian},
    title     = {DepthTrack: Unveiling the Power of RGBD Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10725-10733}
}

@InProceedings{yan2021dot,
  title       = {Depth-only Object Tracking},
  author      = {Yan, Song and Yang, Jinyu and Leonardis, Ales and Kamarainen, Joni-Kristian},
  booktitle   = {Procedings of the British Machine Vision Conference (BMVC)},
  year        = {2021},
  organization= {British Machine Vision Association}
}
```
### Generated LaSOT Depth Images
We manually remove bad sequences, and here are totally 646 sequences (some zip files may be broken, will be updated soon) used the **DenseDepth** method. 
Original DenseDepth outputs are in range [0, 1.0], we multiply 2^16.
Please check [LaSOT](http://vision.cs.stonybrook.edu/~lasot/) for RGB images and groundtruth.

[Download (part01)](https://doi.org/10.5281/zenodo.5482985),
[Download (part02)](https://doi.org/10.5281/zenodo.5484168), 
[Download (part03)](https://doi.org/10.5281/zenodo.5493447),
[Download (part04)](https://doi.org/10.5281/zenodo.5493615),
[Download (part05)](https://doi.org/10.5281/zenodo.5494482),


[Download (part06)](https://doi.org/10.5281/zenodo.5494485),
[Download (part07)](https://doi.org/10.5281/zenodo.5495242),
[Download (part08)](https://doi.org/10.5281/zenodo.5495246),
[Download (part09)](https://doi.org/10.5281/zenodo.5495249),
[Download (part10)](https://doi.org/10.5281/zenodo.5495255)

[Donwload (lion, kangaroo) fix the bad zip files](https://doi.org/10.5281/zenodo.5840345)

[Donwload (pig, rabbit, robot, rubicCube) fix the bad zip files](https://doi.org/10.5281/zenodo.5840339)

[Download (lizard, microphone, monkey, motorcycle, person) fix the bad zip files](https://doi.org/10.5281/zenodo.5840343)

### Generated Got10K Depth Images

[Download (0001 - 0700)](https://doi.org/10.5281/zenodo.5799060), 
[Download (0701 - 1500)](https://doi.org/10.5281/zenodo.5799074), 
[Download (1501 - 2100)](https://doi.org/10.5281/zenodo.5799086),
[Download (2101 - 2600)](https://doi.org/10.5281/zenodo.5799712),

[Downlaod (2601 - 3200)](https://doi.org/10.5281/zenodo.5799718),
[Download (3201 - 3700)](https://doi.org/10.5281/zenodo.5801175),
[Download (3701 - 4000)](https://doi.org/10.5281/zenodo.5801182),
[Download (4001 - 4300)](https://doi.org/10.5281/zenodo.5801711),

[Download (4301 - 4500)](https://doi.org/10.5281/zenodo.5801742),
[Downlaod (4501 - 4800)](https://doi.org/10.5281/zenodo.5803032),
[Download (4801 - 5200)](https://doi.org/10.5281/zenodo.5803036),
[Download (5201 - 5500)](https://doi.org/10.5281/zenodo.5803038),

[Downlaod (5501 - 5800)](https://doi.org/10.5281/zenodo.5803042),
[Download (5801 - 5990)](https://doi.org/10.5281/zenodo.5803366),
[Download (5991 - 6200)](https://doi.org/10.5281/zenodo.5803368),
[Download (6201 - 6400)](https://doi.org/10.5281/zenodo.5803374),

[Downlaod (6401 - 6700)](https://doi.org/10.5281/zenodo.5803379),
[Download (6701 - 7200)](https://doi.org/10.5281/zenodo.5803592),
[Download (7201 - 7600)](https://doi.org/10.5281/zenodo.5803600),
[Download (7601 - 8000)](https://doi.org/10.5281/zenodo.5803604),

[Download (8001 - 8700)](https://doi.org/10.5281/zenodo.5804217),
[Download (8701 - 9000)](https://doi.org/10.5281/zenodo.5804219),
[Download (9001 - 9200)](https://doi.org/10.5281/zenodo.5804221),
[Download (9201 - 9335)](https://doi.org/10.5281/zenodo.5804223)

### Generated COCO Depth Images
[Download](https://doi.org/10.5281/zenodo.5795270)

## How to generate the depth maps for RGB benchmarks
We highly recommend to generate high quality depth data from the existing RGB tracking benchmarks,
such as [LaSOT](http://vision.cs.stonybrook.edu/~lasot/),
[Got10K](http://got-10k.aitestunion.com/),
[TrackingNet](https://tracking-net.org/), and
[COCO](https://cocodataset.org/#home).

We show the examples of generated depth here.
The first row is the results from [HighResDepth](http://yaksoy.github.io/highresdepth/) for LaSOT RGB images,
the second and the third are from [DenseDepth](https://github.com/ialhashim/DenseDepth) for Got10K and COCO RGB images,
the forth row is for the failure cases in which the targets are too close to the background or floor.
The last row is from DenseDepth for CDTB RGB images.



In DeT paper, we used the [DenseDepth](https://github.com/ialhashim/DenseDepth) monocular depth estimation method.
We calculate the Ordinal Error (ORD) on the generated depth for CDTB and our DepthTrack test set, and the mean ORD is about 0.386, which is sufficient for training D or RGBD trackers and we have tested it in our works.

And we also tried the recently [HighResDepth](http://yaksoy.github.io/highresdepth/) from CVPR2021, which also performs very well.

```
@article{alhashim2018high,
  title={High quality monocular depth estimation via transfer learning},
  author={Alhashim, Ibraheem and Wonka, Peter},
  journal={arXiv preprint arXiv:1812.11941},
  year={2018}
}

@inproceedings{miangoleh2021boosting,
  title={Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging},
  author={Miangoleh, S Mahdi H and Dille, Sebastian and Mai, Long and Paris, Sylvain and Aksoy, Yagiz},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9685--9694},
  year={2021}
}
```

### Download
1) Download the training dataset and edit the path in local.py

2) Download the checkpoints for DMT trackers. [code:h0n2](https://pan.baidu.com/s/15IzYQn6CbuJDnYx10Ahcug)


