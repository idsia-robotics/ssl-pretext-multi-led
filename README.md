# Learning to Estimate the Pose of a Peer Robot in a Camera Image by Predicting the State of its LEDs

*Nicholas Carlotti, Mirko Nava, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract

Todo.

<img src="https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_approach.png" width="850" alt="LEDs as Pretext approach" />

Figure 1: *Overview of our approach. A fully convolutional network model is trained to predict the drone position in the current frame by minimizing a loss **L**end defined on a small labeled dataset **T**l (bottom), and the state of the four drone LEDs, by minimizing **L**pretext defined on a large dataset **T**l joined with **T**u (top).*

<br>

Table 1: *Comparison of models, five replicas per row. Pearson Correlation Coefficient ρu and ρv , precision P30 and median of the error D tilde.*

<img src="https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_performance.png" width="900" alt="LEDs as Pretext performance" />

<!--
### Bibtex

```properties
@article{nava2024self,
  author={Nava, Mirko and Carlotti, Nicholas and Crupi, Luca and Palossi, Daniele and Giusti, Alessandro},
  journal={IEEE Robotics and Automation Letters}, 
  title={Self-Supervised Learning of Visual Robot Localization Using LED State Prediction as a Pretext Task}, 
  year={2024},
  volume={9},
  number={4},
  pages={3363-3370},
  doi={10.1109/LRA.2024.3365973},
}
```
-->

### Video

[![Self-Supervised Learning of Visual Robot Localization Using Prediction of LEDs States as a Pretext Task](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_video_preview.gif)](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_video.mp4?raw=true)

### Dataset

The dataset used for our experiments is available [here]().

### Code

The codebase for the approach is avaliable [here](https://github.com/idsia-robotics/ssl-pretext-multi-led/tree/main/code).

##### Requirements

- Python                       3.8.0
- h5py                         3.8.0
- numpy                        1.23.5
- scipy                        1.10.1
- torch                        1.13.1
- torchinfo                    1.8.0
- torchvision                  0.15.2
- tensorboard                  2.12.3
- torch-tb-profiler            0.4.1
- scikit-image                 0.21.0
- scikit-learn                 1.2.2
