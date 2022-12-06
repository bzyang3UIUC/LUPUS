import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt


def save_model(model, model_name):
    model.save(model_name)


def load_model(fl):
    return tf.keras.models.load_model(fl)


def load_data():
    df = pd.read_csv('../data/sensitive_data.csv')
    df_sens = df.drop('name', axis=1)
    vals = tf.convert_to_tensor(df_sens)
    return vals


def discriminator_loss(discriminator, generator, real_vals, noise):
    discriminator_on_real = discriminator(real_vals, training=True)
    # Generate g(z_i)
    generated_sample_vals = generator(noise, training=False)
    discriminator_on_fake = discriminator(generated_sample_vals, training=True)
    return -tf.math.reduce_mean(discriminator_on_real - discriminator_on_fake)


def discriminator_grad(discriminator, generator, real_vals, noise):
    with tf.GradientTape() as tape:
        loss = discriminator_loss(discriminator,
                generator,
                real_vals,
                noise)
    return loss, tape.gradient(loss, discriminator.trainable_variables)


def generator_loss(generator, discriminator, noise):
    generated_sample_vals = generator(noise, training=True)
    loss = discriminator(generated_sample_vals, training=False)
    return -tf.math.reduce_mean(loss)


def generator_grad(generator, discriminator, noise):
    with tf.GradientTape() as tape:
        loss = generator_loss(generator, discriminator, noise)
    return loss, tape.gradient(loss, generator.trainable_variables)


def train(vals):
    vals = load_data()

    # Initialize generator
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='leaky_relu'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(64, activation='leaky_relu'),
        tf.keras.layers.Dense(vals.shape[1], activation='leaky_relu')
        ])

    # Initialize discriminator
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(1, activation='relu')
        ])

    index = tf.range(vals.shape[0])
    sample_num = 128

    # f_w(x_i) - f_w(g(z_i))
    RMSProp_optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=0.01)
    Generator_RMSProp_optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=0.01)

    epochs = 1000
    t2 = 4
    t1 = 1
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        generator_epoch_loss_avg = tf.keras.metrics.Mean()
        for _ in range(t1):
            for _ in range(t2):
                # Sample x_i
                random_index = tf.random.shuffle(index)[:sample_num]
                sample_vals = tf.gather(vals, random_index)

                # Sample z_i
                sample_random = tf.random.normal(sample_vals.shape)

                loss, grads = discriminator_grad(discriminator, generator, sample_vals, sample_random)
                # Add noise to grads here
                # grads = [x + tf.random.normal(x.shape, mean=0.0, stddev=1) for x in grads]
                RMSProp_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

                # Clip gradient here
                # for w in discriminator.trainable_variables:
                #       w.assign(tf.clip_by_value(w, -10, 10))


                epoch_loss_avg.update_state(loss)
        print(f'Discriminator loss: {epoch_loss_avg.result()}')

        # Sample z_i
        sample_random = tf.random.normal(sample_vals.shape)

        loss, grads = generator_grad(generator, discriminator, sample_random)
        Generator_RMSProp_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        generator_epoch_loss_avg.update_state(loss)
        print(f'Generator loss: {generator_epoch_loss_avg.result()}')
    return generator


def validate_synthetic_data(generator, data):
    synthetic_data = generator(tf.random.normal(data.shape))
    df_synth = pd.DataFrame(synthetic_data)
    df_synth.columns = [f'sensitive_feature{x+1}' for x in range(4)]
    df_synth['origin'] = 'synthetic'
    df_real = pd.DataFrame(data)
    df_real.columns = [f'sensitive_feature{x+1}' for x in range(4)]
    df_real['origin'] = 'real'
    df_viz = df_real.append(df_synth)
    for x in range(4):
        sns.histplot(df_viz, x=f'sensitive_feature{x+1}', hue='origin')
        plt.show()


def main():
    data = load_data()
    generator_model = train(data)
    validate_synthetic_data(generator_model, data)
    save_model(generator_model, 'trained_generator')


if __name__ == '__main__':
    main()
