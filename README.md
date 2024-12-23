# It is a small experiment to create an efficient Video Autoencoder for graphics with little VRAM memory and possible use in the [Prometheus](https://github.com/Rivera-ai/Prometheus) model

## The new ```AdaptiveEfficientVideoAutoencoder``` offers a new Video AutoEncoder that can have different qualities or durations, right now tests and improvements are being done on this auto encoder, we have noticed that it takes a long time to learn to rebuild depending on the quality and duration

## All information regarding VideoAutoEncoder usage and training is in the ```Test``` folder

### Memory Usage in ```AdaptiveEfficientVideoAutoencoder``` at 480p 5s videos at 15fps

![](Image/480.png)

### Reconstruction done at 480p 5s 15fps

![](videos/recon_video_480p_60.gif)

## Memory usage in the first version 0.1.0

### RAM
![](Image/RAM.png)

### VRAM
![](Image/VRAM.png)

## Memory usage in the new version 0.2.0

### RAM
![](Image/RAM2.png)

### VRAM
![](Image/VRAM2.png)


## Installation

```bash
git clone https://github.com/Rivera-ai/VideoAutoencoder.git
cd VideoAutoencoder
pip install -e .
```

## Training Results V0.1.0   

### Epoch 0 Reconstruction Progress

The following demonstrations show the reconstruction quality at different steps during the first epoch of training:

#### Step 0
![Step 0 Reconstruction](videos/step0_epoch_.gif)

#### Step 50
![Step 50 Reconstruction](videos/step50_epoch_.gif)

#### Step 100
![Step 100 Reconstruction](videos/step100_epoch_.gif)

#### Step 150
![Step 150 Reconstruction](videos/step150_epoch_.gif)

#### Step 200
![Step 200 Reconstruction](videos/step200_epoch_.gif)

## Training Results V0.2.0   

### Epoch 0 Reconstruction Progress

The following demonstrations show the reconstruction quality at different steps during the first epoch of training:

#### Step 0
![Steps 0 Reconstruction](videos/step0_epoch_0.gif)

#### Step 200
![Steps 200 Reconstruction](videos/step200_epoch_0.gif)

### Epoch 1

#### Step 450
![Steps 450 Reconstruction](videos/step450_epoch_1.gif)

### Epoch 2
#### Step 650
![Steps 650 Reconstruction](videos/step650_epoch_2.gif)

### Epoch 3
#### Step 850
![Steps 850 Reconstruction](videos/step850_epoch_3.gif)

### Epoch 4
#### Step 1050
![Steps 1050 Reconstruction](videos/step1050_epoch_4.gif)

## Obviously, training it on larger datasets and for more epochs will yield better results in terms of reconstruction and version 0.2.0 is much better optimized to train even on 3GB of VRAM but at the cost of requiring more epochs and training steps.