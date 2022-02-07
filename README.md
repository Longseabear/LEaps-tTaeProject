# Ttaeproject

Experimental repository for the tTae project.


# Setup

- setting 
```
pip3 install torch==1.1.0
pip3 install tensorboardX
pip3 install opencv-python
```
- git submodule clone
```
git submodule init
git submodule update
```

- add training weight
```
$ wget "https://drive.google.com/uc?export=download&id=1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4" -O ICCV2019-LearningToPaint/renderer.pkl
$ wget "https://drive.google.com/uc?export=download&id=1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR" -O ICCV2019-LearningToPaint/actor.pkl

$ python3 ICCV2019-LearningToPaint/baseline/test.py --max_step=100 --actor=actor.pkl --renderer=renderer.pkl --img=image/test.png --divide=4
$ ffmpeg -r 10 -f image2 -i output/generated%d.png -s 512x512 -c:v libx264 -pix_fmt yuv420p video.mp4 -q:v 0 -q:a 0
(make a painting process video)
```
