# No Reference Opinion Unaware Quality Assessment of Authentically Distorted Images
Nithin C. Babu, Vignesh Kannan and Rajiv Soundararajan

Official pytorch implementation of the WACV'23 paper "**No Reference Opinion Unaware Quality Assessment of Authentically Distorted Images**".

![Architecture](./imgs/MainArch.jpg)

## Performance evaluation
### Pre-trained weights
Google drive link for pre-trained weights:
- [Model pre-trained on synthetic distortions](https://drive.google.com/file/d/1reM8kTgLPBRNZI9fHylrN-nJ1xX0t0pG/view?usp=share_link)
- [Model finetuned on authentic distortions](https://drive.google.com/file/d/1-eXxqqUJSHK8mcKyVPRnL3_RDy6OihSv/view?usp=share_link)

Copy the files to ```./pre_trained_models/``` folder.
### Setting up pristine patches
Google drive link for pre-selected pristine patches [link](https://drive.google.com/file/d/1TV2tHbzqThRNIOFCZp6tZMNryl5Z0bbS/view?usp=share_link). Copy the downloaded file to the ```dataset_images``` folder as ```./dataset_images/pristine_patches_096_0.75_0.80.hdf5``` .

### Testing code
Sample testing code for evaluating the final model on different authentically distorted datasets.
```
python ./evaluate_model.py --dataset LIVEC --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
python ./evaluate_model.py --dataset KONIQ --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
python ./evaluate_model.py --dataset LIVEFB --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
python ./evaluate_model.py --dataset CID --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
```
