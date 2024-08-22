# concept-nodes
Reimplementation of some ConceptGraphs functionalities. Check out [our website](https://concept-graphs.github.io/) and [the original code](https://github.com/concept-graphs/concept-graphs).

Please note that this codebase does not support the edges presented in the paper.

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
Alternatively, you can create your own `conf/paths/my_paths.yaml` and append `paths=my_paths` to all commands.

## Model Checkpoints
Download the following to ```cache_dir```, as defined in your config.
 * [mobile_sam.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt)
 * [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt)
 * [scannet200_classes.txt](https://raw.githubusercontent.com/concept-graphs/concept-graphs/66175d63f466d264edce9f1fb6987c5ba1dcac0e/conceptgraph/scannet200_classes.txt)

## Example Data
You can download the Replica dataset to `data_dir` using the 
[download script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) from Nice-SLAM.

# Usage
## Quickstart

Mapping configs are defined with [Hydra](https://hydra.cc/docs/intro/). 

We define specific experiments with the `algo` and `dataset` keys. To run the detector variant of Concept-Graphs on a low-resolution
version of the Replica dataset, try

```bash
python3 main.py algo=CGDetector dataset=Replica_low sim_thresh=0.89
```
The map and other assets will be saved to `output_dir`. `main.py` will also create
a symlink to the latest output in `output_dir/latest_map`.

For the slower, SAM-only variant, try `algo=CG` with `sim_thresh=0.85`.

## Parameters

The following arguments can be added to the main command.

* `sim_thresh=0.9`: Similarity threshold between 0 and 1 to merge objects. A high threshold generally gives better object definition, but also a higher occurence of oversegmented or duplicated objects.
* `overlap_eps=0.025`: The radius used for computing the geometric similarity (point cloud overlap). Generally, using the same value as `voxel_size` is a good default.
* `voxel_size=0.025`: The voxel size used for downsampling the object point clouds.
* `denoising_eps=0.1`: The epsilon parameter of the DBSCAN denoising callback. Setting this to 3 or 4 times the `voxel_size` is a good default.
* `max_points_pcd=8000`: The maximum number of points used per object for computing the geometric similarity. A lower number will make the program faster, but the geometric similarity less accurate, especially for large objects.
* `final_min_segments=5`: The minimum number of times an object should be detected to be kept in the map. Helpful to get rid of "noisy" segments that only appear once or twice in the sequence.
* `caption=true`: Ask a VLM to caption the objects. By default, we use OpenAI models and you need an `OPENAI_API_KEY` environment variable for this to work.
* `tag=true`: Ask a VLM to tag the objects. By default, we use OpenAI models and you need an `OPENAI_API_KEY` environment variable for this to work.
* `device=cuda`: Torch device for the perception models and mapping algo.
* `debug=true`: Save additional visualizations of the segmentations.
* `save_map=true`: Save the map. See the [Output](#output) section.
* `seed=123`: Seed.

Additionally, you can also use all the dataset arguments detailed in the [rgbd_dataset README](https://github.com/sachaMorin/rgbd_dataset).

The above list only includes the most common arguments. If you understand [Hydra](https://hydra.cc/docs/intro/), there are a lot more options that you can configure from the CLI. Add `--cfg job` to the main command to visualize the full config.


## Visualizer
To visualize the latest map with Open3D (`output_dir/latest_map`), use
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

## Output
The output consists of the following files and directories:
* `config.yaml`: The full config of this map.
* `clip_features.npy`: The object CLIP features as an `(n_objects, n_dims)` array.
* `point_cloud.pcd`: The complete point cloud.
* `segments_anno.json`: The object annotations following the Scannet++ format. This includes the point indices in the main point cloud, the caption and the tag of every object. Look at [this method](https://github.com/sachaMorin/concept-nodes/blob/791677da1f3de3e007fff0b4bb8f1478d2fe0c61/visualizer.py#L172) for an example of how to load object point clouds.
* `segments`: The top RGB and mask crops for every object. By default, the code keeps the crops with the highest mask resolution.
* `object_viz`: Visualization of the RGB crops, the caption and the tag for every object.
* `debug`: If `debug=true`, additional visualizations of the segmentations.

