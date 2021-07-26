# ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning
This repository contains the PyTorch implementation of the paper 
[ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kordopatis-Zilos_ViSiL_Fine-Grained_Spatio-Temporal_Video_Similarity_Learning_ICCV_2019_paper.pdf). 

## Prerequisites
* Python 3
* PyTorch >= 1.1
* Torchvision >= 0.4

## Getting started

### Installation

* Clone this repo:
```bash
git clone -b pytorch https://github.com/MKLab-ITI/visil visil_pytorch
cd visil_pytorch
```
* You can install all the dependencies by
```bash
pip install -r requirements.txt
```
or
```bash
conda install --file requirements.txt
```

### Video similarity calculation
* Create a file that contains the query videos.
Each line of the file have to contain a video id and a path to the corresponding video file,
 separated by a tab character (\\t). Example:

        wrC_Uqk3juY queries/wrC_Uqk3juY.mp4
        k_NT43aJ_Jw queries/k_NT43aJ_Jw.mp4
        2n30dbPBNKE queries/2n30dbPBNKE.mp4
                                                 ...	
		

* Create a file with the same format for the database videos.

* Run the following command to calculate the similarity between all the query and database videos
```bash
python calculate_similarity.py --query_file queries.txt --database_file database.txt
```

* For faster processing, you can load the query videos to the GPU memory by adding the flag  ```--load_queries```
```bash
python calculate_similarity.py --query_file queries.txt --database_file database.txt --load_queries
```

* The calculated similarities are stored to the file given to the ```--output_file```. The file is in JSON format and
contains a dictionary with every query id as keys, and another dictionary that contains the similarities of the dataset
videos to the corresponding queries as values. See the example below
```bash
    {
      "wrC_Uqk3juY": {
        "KQh6RCW_nAo": 0.716,
        "0q82oQa3upE": 0.300,
          ...},
      "k_NT43aJ_Jw": {
        "-KuR8y1gjJQ": 1.0,
        "Xb19O5Iur44": 0.417,
          ...},
      ....
    }
```

* Add flag `--help` to display the detailed description for the arguments of the similarity calculation script

```
  -h, --help                     show this help message and exit
  --query_file QUERY_FILE        Path to file that contains the query videos (default: None)
  --database_file DATABASE_FILE  Path to file that contains the database videos (default: None)
  --output_file OUTPUT_FILE      Name of the output file. (default: results.json)
  --batch_sz BATCH_SZ            Number of frames contained in each batch during feature extraction. Default: 128 (default: 128)
  --batch_sz_sim BATCH_SZ_SIM    Number of feature tensors in each batch during similarity calculation. (default: 2048)
  --gpu_id GPU_ID                Id of the GPU used. (default: 0)
  --load_queries                 Flag that indicates that the queries will be loaded to the GPU memory. (default: False)
  --workers WORKERS              Number of workers used for video loading. (default: 8)
```

### Evaluation
* We also provide code to reproduce the experiments in the paper.

* First, download the videos of the dataset you want. The supported options are:
    * [FIVR-5K, FIVR-200K](http://ndd.iti.gr/fivr/) - Fine-grained Incident Video Retrieval
    * [CC_WEB_VIDEO](http://vireo.cs.cityu.edu.hk/webvideo/) - Near-Duplicate Video Retrieval 
    * [SVD](https://svdbase.github.io/) - Near-Duplicate Video Retrieval 
    * [EVVE](http://pascal.inrialpes.fr/data/evve/) - Event-based Video Retrieval 

* Determine the pattern based on the video id that the source videos are stored. For example,
 if all dataset videos are stored in a folder with filename the video id and the extension `.mp4`,
 then the pattern is `{id}.mp4`. If each dataset video is stored in a different folder based on their
 video id with filename `video.mp4`, then the pattern us `{id}/video.mp4`.
    * The code replaces the `{id}` string with the id of the videos in the dataset
    * Also, it support supports Unix style pathname pattern expansion. For example, if video files have 
    various extension, then the pattern can be e.g. `{id}/video.*`
    * For FIVR-200K, EVVE, the Youtube ids are considered as the video ids
    * For CC_WEB_VIDEO, video ids derives from the number of the query set that the video belongs to, 
    and the basename of the file. In particular, the video ids are in form `<number_of_query_set>/<basename>`, e.g. `1/1_1_Y`
    * For SVD, video ids derives from the full filename of the videos. In particular, the video ids are in form `<video_filename>.mp4`, e.g. `6520763290014977287.mp4`

* Run the `evaluation.py` by providing the name of the evaluation dataset, the path to video files, 
the pattern that the videos are stored
```
python evaluation.py --dataset FIVR-5K --video_dir /path/to/videos/ --pattern {id}/video.* --load_queries
```

### Use ViSiL in your Python code

Here is a toy example to run ViSiL on any data.

```python
import torch

from model.visil import ViSiL
from utils import load_video

# Load the two videos from the video files
query_video = torch.from_numpy(load_video('/path/to/query/video'))
target_video = torch.from_numpy(load_video('/path/to/target/video'))

# Initialize pretrained ViSiL model
model = ViSiL(pretrained=True).to('cuda')
model.eval()

# Extract features of the two videos
query_features = model.extract_features(query_video.to('cuda'))
target_features = model.extract_features(target_video.to('cuda'))

# Calculate similarity between the two videos
similarity = model.calculate_video_similarity(query_features, target_features)
```

## Related Projects
For improved performance and better computational efficiency, see our **[DnS](https://github.com/mever-team/distill-and-select)** repo.


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

## Contact for further details about the project

Giorgos Kordopatis-Zilos (georgekordopatis@iti.gr)
