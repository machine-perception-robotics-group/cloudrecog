# Cloud_yolo

Requirements:
* chainer
* pytest
* h5py
* ros-kinetic-uvc-camera

```
$ wget http://mprg.cs.chubu.ac.jp/~h_inoko/indexof/cloudrecog/epoch-100.model
$ mv epoch-100.model scripts/yolo/
$ catkin_make
$ source devel/setup.bash

# for server
$ roslaunch cloud_yolo server.launch gpu:=0

# for client
$ roslaunch cloud_yolo client.launch gpu:=-1 video:=/dev/video0 ns:=namespace1
```
