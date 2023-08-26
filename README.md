# <p align="center">No Reference Opinion Unaware Quality Assessment of Authentically Distorted Images</p>
<p align="center">
Nithin C. Babu, Vignesh Kannan, and Rajiv Soundararajan
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=Vgw0ithqHxE"><img src="./imgs/yt_thumb.jpg" width="150"></a>
</p>

<p align="center">
<a>Official pytorch implementation of the WACV'23 paper:</a><br>
<a href="https://openaccess.thecvf.com/content/WACV2023/html/Babu_No_Reference_Opinion_Unaware_Quality_Assessment_of_Authentically_Distorted_Images_WACV_2023_paper.html">No Reference Opinion Unaware Quality Assessment of Authentically Distorted Images</a>

</p>

![Architecture](./imgs/MainArch3.jpg)

## Environment
The code has been tested with:
- Python **3.6.13**
- Pytorch **1.10.1**
- Torchvision **0.11.2**
- CUDA **11.3.1**

Set up an environment with Python **3.6.13** and install all required libraries with the ```requirements.txt``` file. If you use conda, follow the instructions below:

### Setting up conda environment
Execute the following lines one by one on the terminal.
```
conda create --name iqa_content_sep python=3.6.13
conda activate iqa_content_sep
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install --file requirements_extra.txt -c conda-forge
```

## Training
### Synthetic data pretraining
To train the model on the KADIS dataset, run the following:
```
python ./synthetic_pretraining.py --log
```

### Authentic data fine-tuning
After synthetic data pre-training, please run the script below to fine-tune it further on 10k samples of the AVA dataset. The AVA dataset samples used for training are specified in the ```./datasets/AVA_10k_set.csv``` file.
```
python ./authentic_finetuning.py --log --mi_contrastive
```

## Performance evaluation
### Pre-trained weights
Google Drive link for pre-trained weights:
- [Model pre-trained on synthetic distortions](https://drive.google.com/file/d/1reM8kTgLPBRNZI9fHylrN-nJ1xX0t0pG/view?usp=share_link)
- [Model finetuned on authentic distortions](https://drive.google.com/file/d/1-eXxqqUJSHK8mcKyVPRnL3_RDy6OihSv/view?usp=share_link)

Copy the files to the ```./pre_trained_models/``` folder.
### Setting up pristine patches
Google Drive link for pre-selected pristine patches [link](https://drive.google.com/file/d/1TV2tHbzqThRNIOFCZp6tZMNryl5Z0bbS/view?usp=share_link). Copy the downloaded file to the ```dataset_images``` folder as ```./dataset_images/pristine_patches_096_0.75_0.80.hdf5``` .

### Testing code
Sample testing code for evaluating the final model on different authentically distorted datasets.
```
python ./evaluate_model.py --dataset LIVEC --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
python ./evaluate_model.py --dataset KONIQ --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
python ./evaluate_model.py --dataset LIVEFB --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
python ./evaluate_model.py --dataset CID --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
```
## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{iqa_content_sep,
    author    = {Babu, Nithin C. and Kannan, Vignesh and Soundararajan, Rajiv},
    title     = {No Reference Opinion Unaware Quality Assessment of Authentically Distorted Images},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {2459-2468}
}
```
## License
MIT License

Copyright (c) 2023 Nithin C Babu, Vignesh Kannan, Rajiv Soundararajan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

