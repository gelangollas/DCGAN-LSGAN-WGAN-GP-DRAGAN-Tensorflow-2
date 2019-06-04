#
# Fashion-MNIST
#

# DCGAN
python train.py --dataset=fashion_mnist --epoch=100 --adversarial_loss_mode=gan --kep_percent=10
# DRAGAN
python train.py --dataset=fashion_mnist --epoch=100 --adversarial_loss_mode=gan --gradient_penalty_mode=dragan --kep_percent=10
# LSGAN
python train.py --dataset=fashion_mnist --epoch=100 --adversarial_loss_mode=lsgan --kep_percent=10
# WGAN
python train.py --dataset=fashion_mnist --epoch=100 --adversarial_loss_mode=wgan --gradient_penalty_mode=wgan-gp --n_d=5 --kep_percent=10


#
# Cifar10
#

# DCGAN
python train.py --dataset=cifar10 --epoch=100 --adversarial_loss_mode=gan --kep_percent=10
# DRAGAN
python train.py --dataset=cifar10 --epoch=100 --adversarial_loss_mode=gan --gradient_penalty_mode=dragan --kep_percent=10
# LSGAN
python train.py --dataset=cifar10 --epoch=100 --adversarial_loss_mode=lsgan --kep_percent=10
# WGAN
python train.py --dataset=cifar10 --epoch=100 --adversarial_loss_mode=wgan --gradient_penalty_mode=wgan-gp --n_d=5 --kep_percent=10