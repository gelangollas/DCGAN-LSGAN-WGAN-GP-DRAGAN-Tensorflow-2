import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import imlib as im
import module
import pylib as py
import tf2gan as gan
import tf2lib as tl

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line

py.arg('--experiment_name', default='none')
py.arg('--samples_per_class', default=10000, type=int)
args = py.args()
output_folder = Path('output') / args.experiment_name
train_args = py.load_yaml(Path(output_folder / 'settings.yml'))


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# setup dataset
dataset = train_args['dataset']
if dataset == 'cifar10':
    shape = (32, 32, 3)
    n_G_upsamplings = n_D_downsamplings = 3
    n_classes = 10
elif dataset == 'fashion_mnist':
    shape = (32, 32, 1)
    n_G_upsamplings = n_D_downsamplings = 3
    n_classes = 10

# setup the normalization function for discriminator
if train_args['gradient_penalty_mode'] == 'none':
    d_norm = 'batch_norm'
if train_args['gradient_penalty_mode'] in ['dragan', 'wgan-gp']:  # cannot use batch normalization with gradient penalty
    # TODO(Lynn)
    # Layer normalization is more stable than instance normalization here,
    # but instance normalization works in other implementations.
    # Please tell me if you find out the cause.
    d_norm = 'layer_norm'

# networks
output_channels = shape[-1]
batch_size = train_args['batch_size']
G = module.ConvGenerator(input_shape=(1, 1, train_args['z_dim']+n_classes),
                         output_channels=shape[-1],
                         n_upsamplings=n_G_upsamplings,
                         name='G_%s' % train_args['dataset'])
D = module.ConvDiscriminator(input_shape=(shape[0], shape[1], shape[-1]+n_classes),
                             n_downsamplings=n_D_downsamplings,
                             norm=d_norm,
                             name='D_%s' % train_args['dataset'])
G.summary()
D.summary()

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(train_args['adversarial_loss_mode'])

G_optimizer = keras.optimizers.Adam(learning_rate=train_args['lr'], beta_1=train_args['beta_1'])
D_optimizer = keras.optimizers.Adam(learning_rate=train_args['lr'], beta_1=train_args['beta_1'])


@tf.function
def sample(labels_onehot, z):
    return G(tf.concat([z, labels_onehot], axis=-1), training=False)


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G=G,
                                D=D,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_folder, 'checkpoints'),
                           max_to_keep=10)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

generate_summary_writer = tf.summary.create_file_writer(py.join(output_folder, 'summaries', 'generate'))

results_folder = output_folder / 'generated'
py.mkdir(results_folder)

# generating samples of each class
with generate_summary_writer.as_default():
    for i in range(n_classes):
        z = tf.random.normal((args.samples_per_class, 1, 1, train_args['z_dim']))
        labels_onehot = np.zeros((args.samples_per_class, 1, 1, n_classes), dtype=np.float32)
        labels_onehot[:, 0, 0, i] = 1
        x_fake = sample(labels_onehot, z)
        x_fake = np.maximum(x_fake, -1*np.ones(x_fake.shape))
        x_fake = np.minimum(x_fake, np.ones(x_fake.shape))
        img = im.immerge(x_fake, n_rows=1).squeeze()
        with open(results_folder / f'class_{i}.pkl', 'wb') as file:
            pickle.dump(x_fake, file)
