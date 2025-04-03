# NeuroFly: A framework for single neuron reconstruction at whole-brain scale

This package provides tools for semi-automatic neuron reconstruction. Features based on deep learning, like image segmentation and deconvolution are implemented in [tinygrad](https://github.com/tinygrad/tinygrad), which can run on almost any GPU (NVIDIA, AMD, Apple, Qualcomm, Intel).

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/main.png" width="640">

## Update

Our transformer-based autonomous driving model is available now!


Press shortcut 'd' to automatically extend segments.

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/autonomous.gif" width="640">

## Installation

Install the latest version
```
pip install --upgrade git+https://github.com/beanli161514/neurofly.git
```
or
```
pip install neurofly
```

You can also install from [napari hub](https://www.napari-hub.org/plugins/neurofly), using their plugin manager with GUI.

## Dataset

We provide several expert-proofread reconstruction results for testing, model training, and evaluation. [Zenodo Link](https://zenodo.org/records/13328867)


### Content of samples
| name           | size  | species | label type  | imaging |
|----------------|-------|---------|-------------|---------|
| rm009_labeled  | 629MB | macaque | skeleton    | VISoR   |
| mouse_labeled  | 260MB | mouse   | skeleton    | VISoR   |
| z002_labeled   | 204MB | mouse   | skeleton    | VISoR   |
| fmost_labeled  | 370MB | mouse   | skeleton    | fMOST   |
| RM009_noisy_1  | 65MB  | macaque | morphology  | VISoR   |
| RM009_noisy_2  | 65MB  | macaque | morphology  | VISoR   |
| fmost_test     | 65MB  | mouse   | morphology  | fMOST   |
| z002_dendrites | 768MB | mouse   | morphology  | VISoR   |
| RM009_arbor_1  | 288MB | macaque | morphology  | VISoR   |
| RM009_axons_1  | 600MB | macaque | morphology  | VISoR   |
| RM009_axons_2  | 600MB | macaque | morphology  | VISoR   |
| z002           | 8.92G | mouse   | morphology* | VISoR   |

$*$ annotation in progress
### Label format
Morphology labels are graphs saved in SQLite database with 3 tables:
|    segments    |  nodes  |  edges  |
|:--------------:|:-------:|:-------:|
|       sid      |   nid   |   src   |
|     points     |  coord  |   des   |
| sampled_points | creator |   date  |
|                |  status | creator |
|                |   type  |         |
|                |   date  |         |
|                | checked |         |

Segments are results of the segmentation stage, they are used to generate initial nodes and edges.


## Basic usage example

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/pipeline.png" width="640">

NeuroFly packaged 4 napari plugins for image browsing, image segmentation, and data annotation.

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/menu.png" width="320">

### Segmentation

NeuroFly supports whole brain image saved in hierarchical data structures(ims, h5, and zarr) in [Imaris File Format](https://imaris.oxinst.com/support/imaris-file-format) and small image volumes saved in single-channel tiff format. Here we use a mouse brain in our dataset named z002.zarr.zip as example.

This brain is sparsely labeled, which means only a tiny puny part of neurons are lighted and imaged. To extract these foreground singals, you can use the provided command line interface 'seg'. We provide a default weight trained on images captured by VISoR and fMOST.
```
seg -i z002.zarr.zip -vis -d z002.db
```
or use the graphical interface

<img align src="https://github.com/beanli161514/neurofly/raw/main/assets/neuron_seger.png" width="200">


This process may take about 10 hours depending on your you hardware configuration. When finished, you should see the extracted segments and a database file named z002.db in your working dictionary.


An image block with severe contamination and the segmentation result

<img align src="https://github.com/beanli161514/neurofly/raw/main/assets/segmentation.gif" width="640">


### Manual connection and proofreading

Launch annotation tool from napari menu, Plugin -> neurofly -> Segs Annotator

#### Load data
Load image file (z002.zarr.zip) and database file (z002.db), then click **refresh panorama** button to show the panorama view.

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/overall.png" width="600">


#### Select one node as start point of annotation
In panorama mode, you can easily identify sparse, bright signals that are promising for reconstruction. The silde bars 'short segs filter', 'length thres', and 'point size' can be adjusted to hide noise and short segments. 

If you can clearly identify foreground segments, click on one of the conspicuous segments to select it as start point of annotation. Once selected, the id of picked node will be displayed at **node selection**. Then click 'switch mode' to switch to labeling mode, and the tasks will be generated automatically.


#### Task generation
Given a selected node, task generator analyses its connected component and extract all unchecked terminal nodes. The tasks are designed very simple: Connect the center node with the surrounding nodes if there should be an edge. The criterion is whether the edge aligns well with the imaged neuron fibers.

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/task_generation.png" width="480">


#### Node operations
<img src="https://github.com/beanli161514/neurofly/raw/main/assets/labeling_mode.png" width="480">

In each task, a center node and nearby segments are rendered, you can add/remove nodes and edges to get a reasonable local structure.


Left click on nodes to add/remove an edge between it and the center node

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/add_edges.gif" width="480">

Right click to remove a node

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/remove_nodes.gif" width="480">

Press 'g' or use left panel to switch to 'image' layer, then right click to add points

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/add_nodes.gif" width="480">

Use dropdown selection in right panel to add type label for center node.

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/change_type.gif" width="480">

#### Deconvolution

Press 'i' or click on 'deconvolution' to deconvolve the image

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/deconv.gif" width="640">


#### Proofreading

If you find something wrong when labeling, for example, two somas are connected together. You can use proofreading mode to check the neuron branch by branch.
<img src="https://github.com/beanli161514/neurofly/raw/main/assets/proofreading.gif" width="640">



### Performance
NeuroFly supports rendering of more than ten million points. (tested on M3 Macbook Air and RTX 3090 workstation)

<img src="https://github.com/beanli161514/neurofly/raw/main/assets/dense.jpg" width="640">



### Export as swc file
Switch to panorama mode, adjust 'length_thres' to filter out short segments and keep only complete neurons. Then press 'export swc files', each neuron will be saved as one .swc file in your working dictionary.


<img src="https://github.com/beanli161514/neurofly/raw/main/assets/export.jpg" width="640">