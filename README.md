
<div align="center" style="margin-bottom: 0;">
  <img src="https://github.com/mishra-18/VisionGraph/assets/155224614/3dc9e22d-2479-4f69-be7a-d47a4baa134e" width="300">
</div>
<div align="center"><b>Create topology map for image segments</b>


[![PyPI](https://img.shields.io/badge/PyPI-V0.1.1-blue.svg)](https://pypi.org/project/graphvision/)


## GraphVision
A library fascilitating generation of topological representation fpr the image segments. The topological graph is created based on the segments and visual embeddings as nodes. The segment topology, doesn't only keep the spatial but semantic infromation as well, this is very helpful for various tasks in visual robotics[1] and localization. 

GraphVision provides such graphical representations that let us perform visual query, and communicate with our segmentation graph, this would generally require a lot of preprocessing hassle.
VisonGraph does it all itself, and also provides functionalities to visualize the segment topology and perfom visual query on the graph leveraging dijkstras` algorithm for localization.

## Usage
<div align="center" style="margin-bottom: 0;">
  <img src="https://github.com/mishra-18/VisionGraph/assets/155224614/cbd5f159-fa22-4914-85b3-91defae665ff">
</div>

The library requirements are flexible but still its suggested to use a virtual environment.
```python
pip install graphvision
```
Read an image as rgb, import the ```Generator``` and ```SegmentGraph```. Initialize the generator, call the segmentation model, generate the masks and that is it, you are ready create the topological graph for your segments.
```python
from graphvision import Generator, SegmentGraph
gen = Generator()
mask_gen = gen.SAM()
segments = gen.generate_masks(mask_gen, image_rgb)
```
You can use visualize or plot it using ```gen.plot_segments(segments)```

**Create the topological graph**
```python
sg = SegmentGraph(segments, image_rgb)
G, centroids = sg.get_topology(dist_thres=150) # Other Optional Parameters: area_percent, add_to_bbox
```

Two graph ```G``` can be returned as a 'networkx' (default) or as a PyTorch geormetric object. This graphical representation can now be leveraged in various vision tasks such as object localization, environment mapping in robotics based on both spatial and semantic features.

**Query the segment graph**

You can also perform visual queries on the graph, to locate objects dependent on other nodes (neighbouring objects), This a naive implementation, please kindly go through the colab notebook to understand to understand it in greateer detail.
```python
result = sg.query_segment_graph(G, query_pair=("Girl", "plant"), show_legend=True)
```
## Reference
1. [RoboHop: Segment-based Topological Map Representation for Open-World Visual Navigation
](https://oravus.github.io/RoboHop/)
