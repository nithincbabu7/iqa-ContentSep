# No Reference Opinion Unaware Quality Assessment of Authentically Distorted Images
Nithin C. Babu, Vignesh Kannan and Rajiv Soundararajan

Official pytorch implementation of the WACV'23 paper "No Reference Opinion Unaware Quality Assessment of Authentically Distorted Images".

## Performance Evaluation
### Setting up pristine patches
Google drive link for pre-selected pristine patches [link](https://drive.google.com/file/d/1TV2tHbzqThRNIOFCZp6tZMNryl5Z0bbS/view?usp=share_link). Copy the downloaded file to the ```dataset_images``` folder as ```./dataset_images/pristine_patches_096_0.75_0.80.hdf5``` .

### Testing code
Sample testing code on LIVE Challenge dataset.
```
python ./evaluate_model.py --dataset LIVEC --model_weights ./pre_trained_models/auth_ft_cd.pth --eval_result_dir ./results/auth_ft_cd/
```
