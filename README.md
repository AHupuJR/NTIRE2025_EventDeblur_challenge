# [NTIRE 2025 the First Challenge on Event-Based Deblurring](https://codalab.lisn.upsaclay.fr/competitions/21498) @ [CVPR 2025](https://cvlai.net/ntire/2025/)

<p align="center">
  <img src="./figs/logo.jpg" alt="Logo" width="600">
</p>


This is a simple introduction to the dataset and basic codes.

## Downloading Testset
### Blurry images and Raw events
Testset input: 

[CodaLab_downloading_link](https://codalab.lisn.upsaclay.fr/my/datasets/download/3ed362b8-9084-414d-a5f3-d906708773cf); [Kaggle_downloading_link](https://www.kaggle.com/datasets/lei0331/highrev-testset)

(Optional) Voxel input for testset:
[CodaLab_downloading_link](https://codalab.lisn.upsaclay.fr/my/datasets/download/bf89f778-353a-4a51-9dba-69894de81db0); [Kaggle_downloading_link](https://www.kaggle.com/datasets/lei0331/highrev-testset)

## Downloading Trainset and Valset

### Raw events
The [HighREV dataset with raw events](https://codalab.lisn.upsaclay.fr/my/datasets/download/9f275580-9b38-4984-b995-1e59e96b6111)


### Voxel grids
***If you find the data loading too slow***, we also provide processed voxel grid (bin=6) for convenience, which is optional.

[Processed voxel grid of events](https://codalab.lisn.upsaclay.fr/my/datasets/download/c83e95ab-d4e6-4b9f-b7de-e3d3b45356e3)




The structure of the HighREV dataset with raw events is as following:

```
    --HighREV
    |----train
    |    |----blur
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |    |----event
    |    |    |----SEQNAME_%5d_%2d.npz
    |    |    |----...
    |    |----sharp
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |----val
    ...

```
For each blurry image, there are several NPZ files containing events. By concatenating them, the events for the entire exposure time can be obtained. More details please refer to `./basicsr/data/npz_image_dataset.py`


### Converting events to voxel
By using `./basicsr/utils/npz2voxel.py` you can convert raw events to voxel grids by you own offline.

#### Dataset codes:
`./basicsr/data/npz_image_dataset.py` for processing raw events.
`./basicsr/data/voxelnpz_image_dataset.py` for processing voxel grids.




## How to start training?

We provide a simple codebase here:


```
git clone https://github.com/AHupuJR/NTIRE2025_EventDeblur_challenge
cd NTIRE2025_EventDeblur_challenge
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```
Single GPU training:
```
python ./basicsr/train.py -opt options/train/HighREV/EFNet_HighREV_Deblur.yml
```
Multi-GPU training:
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/HighREV/EFNet_HighREV_Deblur.yml --launcher pytorch
```


## How to start testing?
Example:
```
python3 basicsr/test.py -opt options/test/HighREV/EFNet_HighREV_Deblur.yml
```

Calculating flops:
set ``print_flops`` to ``true`` and set your input shapes in ``flops_input_shape`` in the test yml file.
Example:
```
print_flops: true 
flops_input_shape: 
  - [3, 256, 256] # image shape
  - [6, 256, 256] # event shape
```

Be sure to modify the path configurations in yml file.



## Develop your own model
We recommand to used basicsr (already used here, [tutorial](https://github.com/XPixelGroup/BasicSR)) for developing. It is easy to change the models in `./basicsr/models`.



## Use your own codes
The dataset-related code is in `./basicsr/data/npz_image_dataset.py`. If you are not using the code from this repository, please integrate it into your own code for convenience.


## Others
The aim is to obtain a network design / solution that fusing events and images and produce high quality results with the best performance (i.e., PSNR). We suggest using HighREV dataset for training.

For the sake of fairness, please do not train your model with the (HighREV) validation GT images.


The top ranked participants will be awarded and invited to follow the CVPR submission guide for workshops to describe their solution and to submit to the associated NTIRE workshop at CVPR 2025.


## Solution codebases from the attending teams

All team solutions codebases are provided in a single **Google Drive archive**. Download it here: [Google Drive link](https://drive.google.com/file/d/1po89d2IVn-pSKcQZcJKAQnh3HhGNV_Uh/view?usp=sharing). After extracting the archive, you'll find `./codebases/<team_name>` folder structure for each team.


* **One folder per team**  
  Example: `./codebases/IVISLAB/`

* **Self-contained instructions**  
  Every team folder includes its own `README.md` that lists:
  - prerequisites
  - installation or environment setup
  - commands to run the code and reproduce results

## Competition results download
Each team has also submitted pre‑computed results (e.g., model outputs, prediction files) so you can evaluate or compare without re‑running experiments:

- IVISLAB: https://mega.nz/file/qAQARCbY#B8Hyf6tLwSS5N48NxXZoECdbouyBhQt-P7rJltJDHP0 
- MiVideoDeblur: https://drive.google.com/drive/folders/1UE10n176zDZwSlwATEk96awNCdyBxijG
- 404NotFound: https://drive.google.com/file/d/1tN2y7gEJLR770lp4fiRQbH7ZzskiLvVN/view?usp=sharing
- Give it a try:  https://drive.google.com/file/d/1mD7TPu4K8XjVm7-6SdsylZCfIX4OyWQ2/view?usp=sharing
- BUMPTT: wget https://1drv.ms/u/c/ea2a370299726471/EcBi4zfln8dOsmerhs8kr9YBlsAlW3EpZNtbMyrRadB2-g?e=qBDhSd
- WEI wget https://drive.google.com/drive/folders/1bMB8cWuQAXP-7WRdFwT2AzzqmZD6AtC8?usp=sharing
- PixelRevive: https://mail.163.com/large-attachment-download/index.html?p=X-NTES-HUGE-ATTACHMENT&file=djAyY0RsSjlpT2g3eUJ6eDBadzVkZHhmZz09&title=Re%3ARe%3A%20Re%3A%20NTIRE%202025%20Event-Based%20Image%20Deblurring%20Challenge%20-%20PixelRevive%20-%2001
- CHD: https://wx.mail.qq.com/ftn/download?func=3&key=71666466b9cffbd4731c2fdc133906184c125004540156071902550757140006570749020c0a051a07560750540a0000575f5c523529344351151039475c4742581255484f50443709&code=4fdf5947&from=qqmail
- SMU: https://smu-my.sharepoint.com/personal/pmnsbandara_smu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpmnsbandara%5Fsmu%5Fedu%5Fsg%2FDocuments%2FCVPR25%5FDeblurringComp%2FSMU%5Ffinal%5Fsubmission&ga=1
- CMSL: https://drive.google.com/file/d/11M5_ZspBi4mf0h10xNfuHPQxOllFNtB-/view?usp=drive_link
- JNU620: https://drive.google.com/file/d/1F5Qac3Fx_BchsQCJ0YDkxubBLgHLU59p/view?usp=sharing
- Colab: https://drive.google.com/file/d/1kqIfiw3YqwgO9QvdDr3cb7XNJMZBE7dZ/view?usp=drive_link   
- KUnet: wget https://www.dropbox.com/scl/fi/yrky29x2mdwt3k8e40yol/Results.ziprlkey=q0mj8n0i35gsrkl628lzobhlq&st=n16bdckt&dl=0

   
<!-- ## How to add your model to this baseline?
1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1XVa8LIaAURYpPvMf7i-_Yqlzh-JsboG0hvcnp-oI9rs/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
This repository shows how to add noise to synthesize the noisy image. It also shows how you can save an image. -->



## Citations

```
@inproceedings{sun2023event,
  title={Event-based frame interpolation with ad-hoc deblurring},
  author={Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Sun, Peng and Cao, Jiezhang and Zhang, Kai and Jiang, Qi and Wang, Kaiwei and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18043--18052},
  year={2023}
}

@inproceedings{sun2022event,
  title={Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
  author={Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Gool, Luc Van},
  booktitle={European Conference on Computer Vision},
  pages={412--428},
  year={2022},
  organization={Springer}
}
```
