# Object-Detection-with-caffe

Object detection with caffe

Original code of [1]

An spatial image pyramid is computed for the input image and is forward-passed through a convolultional neural network which produces a class probability distribution for each pyramid scale. For each scale the result of the forward pass is the production of a heatmap which indicates the presence or absence of classes for all parts of the input. The heatmaps at all scales are combined using Non Maximum Suppression (NMS) to produce the final bounding boxes containing objects in various sizes.

<p align="center">
<img width="999" height="160" src="https://github.com/danaitri/Object-Detection-with-caffe/blob/master/canvas.png">
</p>

<iframe width="1280" height="528" src="https://www.youtube.com/embed/BxU1MOYiL3Y" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


[![Alt text](https://img.youtube.com/vi/VID/0.jpg)](https://www.youtube.com/watch?v=BxU1MOYiL3Y)

[1]. Danai Triantafyllidou, Paraskevi Nousi, Anastasios Tefas:
Fast Deep Convolutional Face Detection in the Wild Exploiting Hard Sample Mining. Big Data Research 11: 65-76 (2018)


