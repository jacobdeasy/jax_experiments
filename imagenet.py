import flax
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from flax import nn


class ResNetBlock(nn.Module):
    def apply(self, x, filters, *, conv, norm, act, strides=(1, 1)):
        residual = x
        y = conv(x, filters, (3, 3), strides)
        y = norm(y)
        y = act(y)
        y = conv(y, filters, (3, 3))
        y = norm(y, scale_init=nn.initializers.zeros)

        if residual.shape != y.shape:
            residual = conv(residual, filters, (1, 1), strides, name='conv_proj')
            residual = norm(residual, name='norm_proj')

        return act(residual + y)


class BottleneckResNetBlock(nn.Module):
    def apply(self, x, filters, *, conv, norm, act, strides=(1, 1)):
        residual = x
        y = conv(x, filters, (1, 1))
        y = norm(y)
        y = act(y)
        y = conv(y, filters, (3, 3), strides)
        y = norm(y)
        y = act(y)
        y = conv(y, filters * 4, (1, 1))
        y = norm(y, scale_init=nn.initializers.zeros)

        if residual.shape != y.shape:
            residual = conv(residual, filters * 4, (1, 1), strides, name='conv_proj')
            residual = norm(residual, name='norm_proj')

        return act(residual + y)


class ResNet(nn.Module):
    def apply(self, x, num_classes, *, stage_sizes, block_cls, num_filters=64, dtype=jnp.float32, act=nn.relu, train=True):
        conv = nn.Conv.partial(bias=False, dtype=dtype)
        norm = nn.BatchNorm.partial(use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=dtype)

        x = conv(x, num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')
        x = norm(x, name='bn_init')
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = block_cls(x, num_filters * 2 ** i, strides=strides, conv=conv, norm=norm, act=act)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(x, num_classes, dtype=dtype)
        x = jnp.asarray(x, dtype)
        x = nn.log_softmax(x)

        return x


ResNet18 = ResNet.partial(stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = ResNet.partial(stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = ResNet.partial(stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = ResNet.partial(stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = ResNet.partial(stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = ResNet.partial(stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)


@jax.vmap
def cross_entropy_loss(logits, label):
    return -logits[label]


def compute_metrics(logits, labels):
    loss = jnp.mean(cross_entropy_loss(logits, labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    return {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        logits = model(batch['image'])
        loss = jnp.mean(cross_entropy_loss(
            logits, batch['label']))
        return loss
    grad = jax.grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)

    return optimizer


@jax.jit
def eval(model, eval_ds):
    logits = model(eval_ds['image'] / 255.0)  # does this need to be generalised to 3d for Imagenet ?

    return compute_metrics(logits, eval_ds['label'])


def train():
    train_ds = tfds.load('cifar10', split=tfds.Split.TRAIN)
    train_ds = train_ds.map(lambda x: {'image': tf.cast(x['image'], tf.float32),
                                       'label': tf.cast(x['label'], tf.int32)})
    train_ds = train_ds.cache().shuffle(1000).batch(128)
    test_ds = tfds.as_numpy(tfds.load('cifar10', split=tfds.Split.TEST, batch_size=-1))
    test_ds = {'image': test_ds['image'].astype(jnp.float32),
               'label': test_ds['label'].astype(jnp.int32)}

    model = ResNet18
    _, initial_params = model.init_by_shape(
        jax.random.PRNGKey(0),[((1, 32, 32, 3), jnp.float32)])
    model = flax.nn.Model(model, initial_params)
    optimizer = flax.optim.Momentum(
        learning_rate=0.1, beta=0.9).create(model)

    for epoch in range(10):
        for batch in tqdm(tfds.as_numpy(train_ds)):
            batch['image'] = batch['image'] / 255.0
            optimizer = train_step(optimizer, batch)
        metrics = eval(optimizer.target, test_ds)
        print('eval epoch: %d, loss: %.4f, accuracy: %.2f'
            % (epoch+1, metrics['loss'], metrics['accuracy'] * 100))


if __name__ == "__main__":
    train()

