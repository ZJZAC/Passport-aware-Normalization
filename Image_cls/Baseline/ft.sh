python train_v23.py  --arch alexnet  -tf -tb -t all  \
--passport-config  passport_configs/alexnet_passport.json  \
--dataset  cifar100    --tl-dataset  cifar10




python train_v1.py  --arch alexnet  -tf  -t original  \
--dataset  cifar100    --tl-dataset  cifar10

python train_v1.py  --arch resnet  -tf  -t original  \
--passport-config  passport_configs/resnet18_passport.json  \
--dataset  cifar100    --tl-dataset  cifar10