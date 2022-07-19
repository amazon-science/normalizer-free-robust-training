Main reference: [https://github.com/hendrycks/imagenet-r/tree/master/DeepAugment]

## Generate AugMented dataset:
Run the following commands in parallel under `deepaug` folder:

`CUDA_VISIBLE_DEVICES=2 python EDSR_distort_imagenet.py --total_workers 4 --worker_number 0`

`CUDA_VISIBLE_DEVICES=2 python EDSR_distort_imagenet.py --total_workers 4 --worker_number 1`

`CUDA_VISIBLE_DEVICES=2 python EDSR_distort_imagenet.py --total_workers 4 --worker_number 2`

`CUDA_VISIBLE_DEVICES=2 python EDSR_distort_imagenet.py --total_workers 4 --worker_number 3`


`CUDA_VISIBLE_DEVICES=3 python CAE_distort_imagenet.py --total_workers 4 --worker_number 0`

`CUDA_VISIBLE_DEVICES=3 python CAE_distort_imagenet.py --total_workers 4 --worker_number 1`

`CUDA_VISIBLE_DEVICES=3 python CAE_distort_imagenet.py --total_workers 4 --worker_number 2`

`CUDA_VISIBLE_DEVICES=3 python CAE_distort_imagenet.py --total_workers 4 --worker_number 3`

