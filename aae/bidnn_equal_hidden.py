# biDNN stab on keras

# load latent datasets

# set up three layers and make them bidirectional

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Cropping2D
from keras.layers import Concatenate, MaxPooling2D, merge
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
from scipy.stats import bernoulli
from keras.layers import Lambda
from keras import regularizers
from benthic_utils import bathy_data, benthic_img_data
from keras.utils import multi_gpu_model
import tensorflow as tf



class DenseTranspose(Dense):
	"""
	A Keras dense layer that has its weights set to be the transpose of
	another layer. Used for implemeneting BidNNs.
	(needs import tensorflow as tf)
	"""
	def __init__(self, other_layer, **kwargs):
		super().__init__(other_layer.input_dim, **kwargs)
		self.other_layer = other_layer


	def build(self, input_shape):
		assert len(input_shape) >= 2
		input_dim = input_shape[-1]
		self.input_dim = input_dim
		self.input_spec = [InputSpec(dtype=K.floatx(),ndim='2+')]

		self.kernel = tf.transpose(self.other_layer.kernel)

		if self.use_bias:
			self.bias = self.add_weight((self.output_dim,),
									 initializer='zero',
									 name='bias',
									 regularizer=self.bias_regularizer,
									 constraint=self.bias_constraint)
		else:
			self.bias = None

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights
		self.built = True

# from bathy latent to image latent



class BiDNN():
    def __init__(self):
        self.modA_dim = 128
        self.modB_dim = 128
		self.modA_shape = (self.modA_dim)
		self.modB_shape = (self.modB_dim)
        self.latent_dim = 128

        optimizerA = Adam(lr=1e-4, decay=1e-6)


        # Build the encoder / decoder
        self.encoderAB = self.model_encoderAB()
        self.decoderAB = self.model_generatorAB()

		self.encoderBA = self.model_encoderBA()
		self.decoderBA = self.model_generatorBA()

		self.encoderBA_shared = self.model_encoderBA_shared()
		self.decoderBA_shared = self.model_generatorBA_shared()


        # inputs to encoder
        modA = Input(shape=self.modA_shape)
		modB = Input(shape=self.modB_shape)

        # the BiDNN going from modality A to modality B
        encoded_AB = self.encoderAB(modA)
        reconstructed_modB = self.decoderAB(encoded_AB)

		# the BiDNN going from modality B to modality A
        encoded_BA = self.encoderBA(modB)
        reconstructed_modA = self.decoderBA(encoded_BA)

		encoded_BA_shared = self.encoderBA_shared(modB,reconstructed_modB.get_layer("W_zB"))
	    reconstructed_modA_shared = self.decoderBA_shared(encoded_AB,encoded_BA.get_layer("W_Az"))

		# linker network
		diff = merge([mod_AB, mod_BA], mode=lambda (x, y): x - y, output_shape=(3,))
		diff_model = Model(input=[mod_AB, mod_BA], output=diff)
		#print(diff_model.predict([input_a, input_b]))


        # The adversarial_autoencoder model  (stacked generator and discriminator)
        with tf.device('/gpu:0'):
            self.multimodal_bidnn = Model([modA, modB], [reconstructed_modB, reconstructed_modA, diff_model])

        # try using multi_gpu
        #try:
        #    self.adversarial_autoencoder = multi_gpu_model(self.adversarial_autoencoder, cpu_relocation=True)
        #    print("Training autoencoder using multiple GPUs ...")
        #except:
        #    print("Training autoencoder on singe GPU or CPU")

        self.multimodal_bidnn.compile(loss=['mse', 'mse', 'mse'],
            loss_weights=[1, 1, 1],
            optimizer=optimizerA)


    #    print("Autoencoder metrics {}".format(self.adversarial_autoencoder.metrics_names))
	def model_encoderAB(self, output_dim=self.latent_dim, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
		modA = Input(shape=(self.modA_dim,))
		h = modA
		zAB = Dense(output_dim, name="W_Az", activation="sigmoid", kernel_regularizer=reg())(h)
		return Model(modA, zAB)

	def model_decoderAB(self, output_dim=self.modB_dim, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
		zAB = Input(shape=(self.latent_dim,))
		h = zAB
		recB = Dense(output_dim, name="W_zB", activation="sigmoid", kernel_regularizer=reg())(h)
		return Model(zAB, recB)

	# going from modality B to A with tied weights

	def model_encoderBA(self, output_dim=self.latent_dim, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
		modB = Input(shape=(self.modB_dim,))
		h = modB
		zBA = Dense(output_dim, name="W_Bz", activation="sigmoid", kernel_regularizer=reg())(h)
		#zBA = TransponseDense(output_dim, name="W_Bz", activation="sigmoid", kernel_regularizer=reg())(h)
		return Model(modB, zBA)

	def model_decoderBA(self, output_dim=self.modB_dim, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
		zBA = Input(shape=(self.latent_dim,))
		h = zBA
		recA = Dense(output_dim, name="W_zA", activation="sigmoid", kernel_regularizer=reg())(h)
		#recA = TransposeDense(output_dim, name="W_zA", activation="sigmoid", kernel_regularizer=reg())(h)
		return Model(zBA, recA)


	def model_encoderBA_shared(self, other_layer, output_dim=self.latent_dim, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
		modB = Input(shape=(self.modB_dim,))
		h = modB
		zBA = TransponseDense(other_layer, output_dim, name="W_Bz", activation="sigmoid", kernel_regularizer=reg())(h)
		return Model(modB, zBA)

	def model_decoderBA_shared(self, other_layer, output_dim=self.modB_dim, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
		zBA = Input(shape=(self.latent_dim,))
		h = zBA
		recA = TransposeDense(other_layer, output_dim, name="W_zA", activation="sigmoid", kernel_regularizer=reg())(h)
		return Model(zBA, recA)



	def train(self, epochs, batch_size=128, sample_interval=50):

		# Load the dataset
		cached_bpatches_latent = '/data/bathy_training/cache_bpatches_ohara_07.npz'
		cached_images_latent = '/data/bathy_training/cache_bpatches_ohara_07.npz'


		(Xbathy_train,_) = load_cached_training_npz(cached_bpatches_latent)
		(Ximg_train,_) = load_cached_training_npz(cached_images_latent)

    	print("shape Ximg_train {}".format(Ximg_train.shape))
		print("shape Xbathy_train {}".format(Xbathy_train.shape))

		print(Ximg_train.shape[0], 'train img samples')
		print(Xbathy_train.shape[0], 'train bathy samples')

		#desc_batch = int(batch_size / 2)
		desc_batch = int(batch_size)

		# plotting metrics
		g_loss_hist = []

		for epoch in range(epochs):

			# ---------------------
			#  Train Generator
			# ---------------------

			# Select a random batch of images
			idx = np.random.randint(0, Xbathy_train.shape[0], batch_size)

			bpatchs = Xbathy_train[idx]
			imgs = Ximg_train[idx]

			# Train the generator
			g_loss = self.multimodal_bidnn.train_on_batch([imgs, bpatchs], [bpatchs, bpatchs_means, np.zeros((batch_size, 1))])

			# Plot the progress
			print ("bathy %d [G loss: %f, mseBI: %f, mseIB: %f, mseZ: %f]" % (epoch, g_loss[0], g_loss[1], g_loss[2]))

			g_loss_hist.append((g_loss[0],g_loss[1],g_loss[2]))

			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				#self.sample_images(epoch)
				#self.sample_autoencoder(epoch, bpatchs, bpatchs_means,"bathy_aae")
				self.sample_metrics(g_loss_hist)

		#save params once done training
		#self.save_model()
		#self.save_latentspace([Xbathy_train,Xbathy_train_means],"z_bathy")

	def sample_metrics(g_loss_hist):
		# plotting the metrics
		plt.plot(g_loss_hist,linewidth=0.5)
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['BI','IB','Zdelta'], loc='center right')
		plt.show()
		plt.savefig("metrics_bidnnl_bathy/aae_bathy_metrics.png")
		plt.close()

	def sample_images(self, epoch):
		r, c = 4, 5

		z = np.random.normal(size=(r*c, self.latent_dim))

		gen_bpatchs = self.decoder_bathy.predict(z)
		gen_bpatchs_means = self.decoder_bathy_mean.predict(z)
		#print("shape gen imgs {}".format(gen_imgs.shape))
		#print("shape gen bpatch {}".format(gen_bpatchs.shape))
		# where does this come from?
		#gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_bpatchs[cnt,:,:,0])
				axs[i,j].set_title("d %.1f" % (100*gen_bpatchs_means[cnt]), color='black')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("bathy_aae_generator/benthic_%d.png" % epoch)
		plt.close()

	def sample_autoencoder(self, epoch,bpatchs, bpatchs_means, save_folder):
		r, c = 4, 2
		namps = r*c

		# Select a random set of images
		#idx = np.random.randint(0, X_train.shape[0], nsamps)
		#imgs = X_train[idx]
		gen_bpatchs, gen_bpatchs_means, valids = self.adversarial_autoencoder.predict([bpatchs,bpatchs_means])

		#gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c*2)
		cnt = 0
		for i in range(r):
			for j in range(c):

				axs[i,2*j].imshow(bpatchs[cnt,:,:,0])
				axs[i,2*j].set_title("d %.1f" % (100*bpatchs_means[cnt]), color='black')
				axs[i,2*j].axis('off')
				axs[i,2*j+1].imshow(gen_bpatchs[cnt,:,:,0])
				axs[i,2*j+1].set_title("d %.1f" % (100*gen_bpatchs_means[cnt]), color='black')
				axs[i,2*j+1].axis('off')
				cnt += 1
		fig.savefig(save_folder+"/benthic_%d.png" % epoch)
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

		save(self.adversarial_autoencoder, "bathy_aae")
		save(self.encoder, "bathy_encoder")
		save(self.decoder_bathy, "bathy_decoder")
		save(self.decoder_bathy_mean, "bathy_decoder_mean")
		save(self.discriminator, "bathy_aae_discriminator")

	def save_latentspace(self, inputdata, latent_name):

		z = self.encoder.predict(inputdata)
		np.save("saved_latent/%s.npy" % latent_name, z)

if __name__ == '__main__':
mmb = BiDNN()
mmb.train(epochs=200000, batch_size=32, sample_interval=5000000)
