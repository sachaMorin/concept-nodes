# concept-nodes
Reimplementation of some ConceptGraphs functionalities. 

# Install
Clone repository and submodules
```bash
git clone --recurse-submodules https://github.com/sachaMorin/concept-nodes.git
```

Install dependencies. Preferably in a virtual environment.
```bash
pip install --upgrade pip
```
```bash
pip install -e concept-nodes
```
```bash
pip install -e concept-nodes/rgbd_dataset
```
## Paths 
Update `conf/paths/paths.yaml`.

```yaml
# @package _global_
cache_dir: ??? # Where to save model checkpoints and other assets
data_dir: ??? # Where to look for datasets by default
output_dir: ??? # Where to save outputs
```
## Model Checkpoints
Download the following to ```cache_dir```, as defined in your config.
 * [mobile_sam.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt)
 * [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt)
 * [scannet200_classes.txt](https://github.com/concept-graphs/concept-graphs/blob/66175d63f466d264edce9f1fb6987c5ba1dcac0e/conceptgraph/scannet200_classes.txt)

## Data
You can download the Replica dataset to `data_dir` using the 
[download script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) from Nice-SLAM.

# Usage
## Mapping

Mapping configs are defined with [Hydra](https://hydra.cc/docs/intro/). We define specific configs as `experiments`.

To run the experiment defined in `conf/experiments/replica.yaml`, run:


```bash
pyhon3 main.py experiment=replica
```
The map and other assets will be saved to `output_dir`. `main.py` will also create
a symlink to the latest output in `output_dir/latest_map.pkl`.

## Hydra

To visualize the entire config, run:
```bash
pyhon3 main.py experiment=replica --cfg job
```
You can override all parameters from the command line. For example, to run the same experiment using
only Mobile-SAM on the first 100 frames, run:
```bash
pyhon3 main.py experiment=replica segmentation=GridMobileSAM dataset.sequence_end=100
```

## Visualizer
To visualize the latest map with Open3D (`output_dir/latest_map.pkl`), use
```bash
python3 visualizer.py
```
or provide your own `$MAP_PATH`
```bash
python3 visualizer.py map_path=$MAP_PATH
```

Various options and colorings are available in the panel
at the bottom right of the window. Some options such as CLIP
queries require to interact with the terminal used to 
launch the visualizer.

