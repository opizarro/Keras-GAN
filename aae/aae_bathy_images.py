from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib as mpl
mpl.use('Agg') # does use DISPLAY
import matplotlib.pyplot as plt

import numpy as np
# added by OP
from keras.layers import Lambda
from keras import regularizers
from benthic_utils import benthoz_data
from keras.utils import multi_gpu_model
import tensorflow as tf

class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        self.bathy_rows = 21
        self.bathy_cols = 21
        self.bathy_shape = (self.bathy_rows,self.bathy_cols)

        optimizer = Adam(lr=2e-4, decay=1e-6)

        # Build and compile the discriminator
        #self.discriminator = self.build_discriminator()
        with tf.device('/gpu:1'):
            self.discriminator = self.model_discriminator()

        # try using multi_gpu
       # try:
        #    self.discriminator = multi_gpu_model(self.discriminator, cpu_relocation=True)
        #    print("Training discriminator using multiple GPUs ...")
        #except:
        #    print("Training discriminator on singe GPU or CPU")

        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.model_encoder()
        self.decoder = self.model_generator()
        self.decoder_bathy = self.model_generator_bathy()

        # inputs to encoder
        img = Input(shape=self.img_shape)
        bpatch = Input(shape=self.bathy_shape)

        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)
        reconstructed_bathy = self.decoder_bathy(encoded_repr)

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False
        # The adversarial_autoencoder model  (stacked generator and discriminator)
        with tf.device('/gpu:0'):
            self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])

        # try using multi_gpu
        #try:
        #    self.adversarial_autoencoder = multi_gpu_model(self.adversarial_autoencoder, cpu_relocation=True)
        #    print("Training autoencoder using multiple GPUs ...")
        #except:
        #    print("Training autoencoder on singe GPU or CPU")


        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[1e3, 1e-1],
            optimizer=optimizer)


    def model_encoder(self, units=512, reg=lambda: regularizers.l1_l2(l1=1e-7, l2=1e-7), dropout=0.5):
        k = 5
        x = Input(shape=self.img_shape)
        h = Conv2D(units// 4, (k, k), padding='same', kernel_regularizer=reg())(x)
        # h = SpatialDropout2D(dropout)(h)
        h = MaxPooling2D(pool_size=(2, 2))(h) # 32 x 32
        #h = LeakyReLU(0.2)(h)
        h = PReLU()(h)
        h = Conv2D(units // 2, (k, k), padding='same', kernel_regularizer=reg())(h)
        # h = SpatialDropout2D(dropout)(h)
        h = MaxPooling2D(pool_size=(2, 2))(h) # 16 x 16
        #h = LeakyReLU(0.2)(h)
        h = PReLU()(h)
        h = Conv2D(units // 2, (k, k), padding='same', kernel_regularizer=reg())(h)
        # h = SpatialDropout2D(dropout)(h)
        h = MaxPooling2D(pool_size=(2, 2))(h) # 8 x 8
        #h = LeakyReLU(0.2)(h)
        h = PReLU()(h)
        h = Conv2D(units, (k, k), padding='same', kernel_regularizer=reg())(h)
        # h = SpatialDropout2D(dropout)(h)
        #h = LeakyReLU(0.2)(h)
        h = PReLU()(h)
        h = Flatten()(h)

        x2 = Input(shape=self.bathy_shape)
        x = ZeroPadding2D((6,5,6,5))(x2) # from 21x21 to 32 x 32
        h2 = Conv2D(units// 4, (k, k), padding='same', kernel_regularizer=reg())(x2)
        # h = SpatialDropout2D(dropout)(h)
        h2 = MaxPooling2D(pool_size=(2, 2))(h2) # 16 x 16
        #h = LeakyReLU(0.2)(h)
        h2 = PReLU()(h2)
        h2 = Conv2D(units // 2, (k, k), padding='same', kernel_regularizer=reg())(h2)
        # h = SpatialDropout2D(dropout)(h)
        h2 = MaxPooling2D(pool_size=(2, 2))(h2) # 8 x 8
        #h = LeakyReLU(0.2)(h)
        h2 = PReLU()(h2)
        h2 = Conv2D(units // 2, (k, k), padding='same', kernel_regularizer=reg())(h2)
        # h = SpatialDropout2D(dropout)(h)
        h2 = MaxPooling2D(pool_size=(2, 2))(h2) # 4 x 4
        #h = LeakyReLU(0.2)(h)
        h2 = PReLU()(h2)
        h2 = Conv2D(units, (k, k), padding='same', kernel_regularizer=reg())(h2)
        # h = SpatialDropout2D(dropout)(h)
        #h = LeakyReLU(0.2)(h)
        h2 = PReLU()(h2)
        h2 = Flatten()(h2)

        hcomb = Concatenate()([h, h2])

        mu = Dense(self.latent_dim, name="encoder_mu", kernel_regularizer=reg())(hcomb)
        log_sigma_sq = Dense(self.latent_dim, name="encoder_log_sigma_sq", kernel_regularizer=reg())(h)
        # z = Lambda(lambda (_mu, _lss): _mu + K.random_normal(K.shape(_mu)) * K.exp(_lss / 2),output_shape=lambda (_mu, _lss): _mu)([mu, log_sigma_sq])
        z = Lambda(lambda ml: ml[0] + K.random_normal(K.shape(ml[0])) * K.exp(ml[1] / 2),
                   output_shape=lambda ml: ml[0])([mu, log_sigma_sq])

        return Model([x,x2], z, name="encoder")




    def model_generator(self, units=512, dropout=0.5, reg=lambda: regularizers.l1_l2(l1=1e-7, l2=1e-7)):
        decoder = Sequential(name="decoder")
        h = 5

        decoder.add(Dense(units * 4 * 4 , input_dim=self.latent_dim, kernel_regularizer=reg()))
        # check channel order on below
        decoder.add(Reshape((4,4,units)))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(Conv2D(units // 2, (h, h), padding='same', kernel_regularizer=reg()))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(UpSampling2D(size=(2, 2)))
        decoder.add(Conv2D(units // 2, (h, h), padding='same', kernel_regularizer=reg()))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(UpSampling2D(size=(2, 2)))
        decoder.add(Conv2D(units // 4, (h, h), padding='same', kernel_regularizer=reg()))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(UpSampling2D(size=(2, 2)))
        decoder.add(Conv2D(3, (h, h), padding='same', kernel_regularizer=reg()))
        # add one more PReLU for fine scale detail?


        # added another upsampling step to get to 64 x 64
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(UpSampling2D(size=(2, 2)))
        decoder.add(Conv2D(3, (h, h), padding='same', kernel_regularizer=reg()))

        decoder.add(Activation('sigmoid'))

        #decoder.summary()
        # above assumes a particular output dimension, instead try below
        #decoder.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
        #decoder.add(Reshape(self.img_shape))

        decoder.summary()

        z = Input(shape=(self.latent_dim,))
        img = decoder(z)

        return Model(z, img)


def model_generator_bathy(self, units=512, dropout=0.5, reg=lambda: regularizers.l1_l2(l1=1e-7, l2=1e-7)):
        decoder = Sequential(name="decoder")
        h = 5

        decoder.add(Dense(units * 4 * 4 , input_dim=self.latent_dim, kernel_regularizer=reg()))
        # check channel order on below
        decoder.add(Reshape((4,4,units)))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(Conv2D(units // 2, (h, h), padding='same', kernel_regularizer=reg()))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(UpSampling2D(size=(2, 2)))
        decoder.add(Conv2D(units // 2, (h, h), padding='same', kernel_regularizer=reg()))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(UpSampling2D(size=(2, 2)))
        decoder.add(Conv2D(units // 4, (h, h), padding='same', kernel_regularizer=reg()))
        # decoder.add(SpatialDropout2D(dropout))
        #decoder.add(LeakyReLU(0.2))
        decoder.add(PReLU())
        decoder.add(UpSampling2D(size=(2, 2))) # 32 x 32
        decoder.add(Conv2D(1, (h, h), padding='same', kernel_regularizer=reg()))


        decoder.add(Activation('sigmoid'))

        # hack to bring back to size of bathymetry 21x21
        decoder.add(Cropping2D((6,5)(6,5))


        #decoder.summary()
        # above assumes a particular output dimension, instead try below
        #decoder.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
        #decoder.add(Reshape(self.img_shape))

        decoder.summary()

        z = Input(shape=(self.latent_dim,))
        bpatch = decoder(z)

        return Model(z, bpatch)


    def model_discriminator(self, output_dim=1, units=512, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
        z = Input(shape=(self.latent_dim,))
        h = z
        mode = 1
        h = Dense(units, name="discriminator_h1", kernel_regularizer=reg())(h)
        # h = BatchNormalization(mode=mode)(h)
        #h = LeakyReLU(0.2)(h)
        h = PReLU()(h)
        h = Dense(units // 2, name="discriminator_h2", kernel_regularizer=reg())(h)
        # h = BatchNormalization(mode=mode)(h)
        #h = LeakyReLU(0.2)(h)
        h = PReLU()(h)
        h = Dense(units // 2, name="discriminator_h3", kernel_regularizer=reg())(h)
        # h = BatchNormalization(mode=mode)(h)
        #h = LeakyReLU(0.2)(h)
        h = PReLU()(h)
        y = Dense(output_dim, name="discriminator_y", activation="sigmoid", kernel_regularizer=reg())(h)
        return Model(z, y)



    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        (Ximg_train,_) = benthic_img_dat()
        (Xbathy_train,_) = bathy_data()

        print("shape Ximg_train {}".format(Ximg_train.shape))
        print("shape Xbathy_train {}".format(Xbathy_train.shape))

        print(Ximg_train.shape[0], 'train img samples')
        print(Xbathy_train.shape[0], 'train bathy samples')
        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3) # only used if image is 2D without channel info

        #desc_batch = int(batch_size / 2)
        desc_batch = int(batch_size)

        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, Ximg_train.shape[0], desc_batch)
            imgs = Ximg_train[idx]
            bpatchs = Xbathy_train[idx]
            #print("shape imgs {}".format(imgs.shape))
            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(desc_batch, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, np.ones((desc_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, np.zeros((desc_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, np.ones((batch_size, 1))])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.sample_autoencoder(epoch, imgs)

    def sample_images(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.decoder.predict(z)
        # where does this come from?
        #gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images_generator/benthic_%d.png" % epoch)
        plt.close()

    def sample_autoencoder(self, epoch,imgs):
        r, c = 5, 2
        namps = r*c

        # Select a random set of images
        #idx = np.random.randint(0, X_train.shape[0], nsamps)
        #imgs = X_train[idx]
        gen_imgs, valids = self.adversarial_autoencoder.predict(imgs)

        #gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c*2)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,2*j].imshow(imgs[cnt])
                axs[i,2*j].axis('off')
                axs[i,2*j+1].imshow(gen_imgs[cnt])
                axs[i,2*j+1].axis('off')
                cnt += 1
        fig.savefig("images_autoenc/benthic_%d.png" % epoch)
        plt.close()



    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "aae_generator")
        save(self.discriminator, "aae_discriminator")


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=1000000, batch_size=32, sample_interval=500)
