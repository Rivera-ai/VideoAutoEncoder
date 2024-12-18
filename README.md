# It is a small experiment to create an efficient Video Autoencoder for graphics with little VRAM memory and possible use in the [Prometheus](https://github.com/Rivera-ai/Prometheus) model

## Memory Usage

### RAM
![](Image/RAM.png)

### VRAM
![](Image/VRAM.png)

## Installation

```bash
git clone https://github.com/Rivera-ai/VideoAutoencoder.git
cd VideoAutoencoder
pip install -e .
```

## Training Results

### Epoch 0 Reconstruction Progress

The following demonstrations show the reconstruction quality at different steps during the first epoch of training:

#### Step 0
![Step 0 Reconstruction](videos/step0_epoch_0.gif)

#### Step 50
![Step 50 Reconstruction](videos/step50_epoch_0.gif)

#### Step 100
![Step 100 Reconstruction](videos/step100_epoch_0.gif)

#### Step 150
![Step 150 Reconstruction](videos/step150_epoch_0.gif)

#### Step 200
![Step 200 Reconstruction](videos/step200_epoch_0.gif)