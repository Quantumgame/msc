# msc
Relevant files from my MSc project

Steps to run a simulation:

### Disclaimer: 
I have not tested whether more steps are required to successfully run a simulation. Let me know if you encounter any issues: 
felipeduque@riseup.net

I have only tested under Arch Linux operating system. It should run fine under other Linux flavors though.

### 1- Install the required python libraries:

numpy
Pillow (PIL)
scipy
multiprocessing
itertools
caffe
opencv (cv2)

### 2- Install V-REP

The student version is free and GPL'd. [Download it here](http://www.coppeliarobotics.com/downloads.html).

### 3- Install Matlab runtime

Unfortunately, third-party EdgeBoxes was written in Matlab. Fortunately, you only need to download the runtime 
environment (which is free) to run the compiled code. [Download it here](https://www.mathworks.com/products/compiler/mcr.html). 
The code was compiled in Matlab 2014b.

### 4- Include lib and Caffe installation directory in $PYTHONPATH

You can do that by
```
export PYTHONPATH=/path/to/lib:/path/to/caffe
```

You can make it persistent by adding it to your `.bashrc` file.

### 5- Run utils/edges-master/run\_my\_edgeboxes.sh

You can run it by 
```
./run\_my\_edgeboxes.sh ~/matlab ../../code/imgs/temp/image_file_name_bbs.txt models/forest/modelBsds.mat
```

where `~/matlab` is the Matlab runtime environment installation directory.
This application periodically checks for changes in the `.txt` file. The main application, `code/code\_10.py`, 
is responsible for making the changes. Thus the application must be running at the same time as the main 
application.

### 6- Open V-REP, load the scene and start the simulation

You must open V-REP inside `code`. The suggested scene is in `code/scenes`.

Start the simulation in V-REP.

### 7- Run the main application

You can run the main application by
```
python code\_10.py sgng14.npy dicionary14.npy 0.1
```
if you want a pre-trained robot. If you want a naive one, run it by 
```
python code\_10.py 0.1
```


