# ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning
This repository contains the Tensorflow implementation of the paper 
[ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kordopatis-Zilos_ViSiL_Fine-Grained_Spatio-Temporal_Video_Similarity_Learning_ICCV_2019_paper.pdf). 
It provides code for the calculation of similarities between the query and database videos provided by the user. 

<img src="https://raw.githubusercontent.com/MKLab-ITI/visil/master/video_similarity.png" width="70%">

## Prerequisites
* Python 3
* Tensorflow CPU or GPU version

## Getting started

### Installation

* Clone this repo:
```bash
git clone https://github.com/MKLab-ITI/visil
cd visil
```
* You can install all the dependencies by
```bash
pip install -r requirements.txt
```
or
```bash
conda install --file requirements.txt
```

* Download and unzip the pretrained model:
```bash
wget ndd.iti.gr/visil/model.zip
unzip model.zip
```

### Video similarity calculation
* Create a file that contains the query videos.
Each line of the file have to contains a video id and a path to the corresponding video file,
 separated by a tab character (\\t). Example:

        wrC_Uqk3juY queries/wrC_Uqk3juY.mp4
        k_NT43aJ_Jw queries/k_NT43aJ_Jw.mp4
        2n30dbPBNKE queries/2n30dbPBNKE.mp4
                                                 ...	
		

* Create a file with the same format for the database videos videos.

* Run the following command to calculate the similarity between all the query and database videos
```bash
python calculate_similarity.py --query_file queries.txt --database_file database.txt --model_dir model/
```

* For faster processing, you can load the query videos to the GPU memory by adding the flag  ```--load_queries```
```bash
python calculate_similarity.py --query_file queries.txt --database_file database.txt --model_dir model/ --load_queries
```

* The calculated similarities are stored to the file given to the ```--output_file```. The file is in JSON format and
contain a dictionary with every query id as key and value another dictionary that contains the similarities of the 
dataset videos to the corresponding query. See the example below
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
```


## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{kordopatis2019visil,
  title={ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning},
  author={Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Ioannis},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
}
```
## Related Projects
**[FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K)**

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

## Contact for further details about the project

Giorgos Kordopatis-Zilos (georgekordopatis@iti.gr) <br>
Symeon Papadopoulos (papadop@iti.gr)
