
<div align="center" style="margin-bottom: 0;">
  <img src="https://github.com/mishra-18/VisionGraph/assets/155224614/3dc9e22d-2479-4f69-be7a-d47a4baa134e" width="300">
</div>
<div align="center"><b>Create topology map for image segments</b>

[![PyPI](https://img.shields.io/badge/pypi-V0.1.2-blue.svg)](https://pypi.org/project/visiongraph/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mErg7NDG8XlnrVJdVjtwzkPd-TQgAhTy?usp=sharing)
</div>


## GraphVision
A library facilitating the generation of topological representations for image segments. The topological graph is created based on segments and visual embeddings as nodes. This segment topology retains not only spatial but also semantic information, making it very useful for various tasks in visual robotics[1] and localization.

GraphVision provides graphical representations that enable us to perform visual queries and interact with our segmentation graph without the usual preprocessing hassle. VisionGraph handles everything and also offers functionalities to visualize the segment topology and perform visual queries on the graph, leveraging Dijkstra's algorithm for localization..

## Usage
<div align="center" style="margin-bottom: 0;">
  <img src="https://github.com/mishra-18/VisionGraph/assets/155224614/cbd5f159-fa22-4914-85b3-91defae665ff">
</div>

The library requirements are flexible but still its suggested to use a virtual environment.
```python
pip install graphvision
```
Read an Image
```python
image_bgr = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```
Initialize the generator, call the segmentation model, generate the masks and that is it! You are ready create the topological graph for your segments.
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

The graph ```G``` can be returned as a networkx (default) or as a PyTorch geometric object. This graphical representation can now be leveraged in various vision tasks such as object localization and environment mapping in robotics, based on both spatial and semantic features..

**Query the segment graph**

You can also perform visual queries on the graph to locate objects dependent on other nodes (neighbouring objects). This a naive implementation, so please go through the [Colab notebook](https://colab.research.google.com/drive/1mErg7NDG8XlnrVJdVjtwzkPd-TQgAhTy?usp=sharing) to understand to understand it in greateer detail.
```python
result = sg.query_segment_graph(G, query_pair=("Girl", "plant"), show_legend=True)
```
## Reference
1. [RoboHop: Segment-based Topological Map Representation for Open-World Visual Navigation
](https://oravus.github.io/RoboHop/)
