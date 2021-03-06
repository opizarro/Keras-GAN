# biDNN stab on keras

# load latent datasets

# set up three layers and make them bidirectional
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


        # inputs to encoder
        modA = Input(shape=self.modA_shape)
		modB = Input(shape=self.modB_shape)

        # the BiDNN going from modality A to modality B
        encoded_AB = self.encoderAB(modA)
        reconstructed_modB = self.decoderAB(encoded_AB)

		# the BiDNN going from modality B to modality A
        encoded_BA = self.encoderBA(modB)
        reconstructed_modA = self.decoderBA(encoded_BA)


        # The adversarial_autoencoder model  (stacked generator and discriminator)
        with tf.device('/gpu:0'):
            self.multimodal_bidnn = Model([modA, modB], [reconstructed_modB, reconstructed_modA])

        # try using multi_gpu
        #try:
        #    self.adversarial_autoencoder = multi_gpu_model(self.adversarial_autoencoder, cpu_relocation=True)
        #    print("Training autoencoder using multiple GPUs ...")
        #except:
        #    print("Training autoencoder on singe GPU or CPU")

        self.multimodal_bidnn.compile(loss=['mse', 'mse'],
            loss_weights=[1, 1],
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
	zBA = TransponseDense(output_dim, name="W_Bz", activation="sigmoid", kernel_regularizer=reg())(h)
	return Model(modB, zBA)

def model_decoderBA(self, output_dim=self.modB_dim, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
	zBA = Input(shape=(self.latent_dim,))
	h = zBA
	recA = TransposeDense(output_dim, name="W_zA", activation="sigmoid", kernel_regularizer=reg())(h)
	return Model(zBA, recA)


def train(self, epochs, batch_size=128, sample_interval=50):

	# Load the dataset
	#(X_train, _), (_, _) = mnist.load_data()


	(Xbathy_train,_) = bathy_data()


	print("shape Xbathy_train {}".format(Xbathy_train.shape))


	print(Xbathy_train.shape[0], 'train bathy samples')
	# Rescale -1 to 1
	#X_train = (X_train.astype(np.float32) - 127.5) / 127.5
	#X_train = np.expand_dims(X_train, axis=3) # only used if image is 2D without channel info
#HACK with 100 scale for mean depth
	# remove mean from depth and use as separate input
	Xbathy_train_means = 0.01*np.mean(Xbathy_train,axis=(1,2))
	print("shape Xbathy_train_means ", Xbathy_train_means.shape)

	for k in np.arange(Xbathy_train.shape[0]):
		Xbathy_train[k,:,:,0] = Xbathy_train[k,:,:,0] - Xbathy_train_means[k]*100


	#desc_batch = int(batch_size / 2)
	desc_batch = int(batch_size)

	#noise_frac = 0.05
	#missing_prob = 0.5

	# plotting metrics
	d_loss_hist = []
	g_loss_hist = []

	for epoch in range(epochs):


		# ---------------------
		#  Train Discriminator
		# ---------------------

		# Select a random half batch of images
		idx = np.random.randint(0, Xbathy_train.shape[0], desc_batch)

		bpatchs = Xbathy_train[idx]
		bpatchs_means = Xbathy_train_means[idx]

		#print("shape imgs {}".format(imgs.shape))

		latent_fake = self.encoder.predict([bpatchs,bpatchs_means])
		latent_real = np.random.normal(size=(desc_batch, self.latent_dim))

		# Train the discriminator


		d_loss_real = self.discriminator.train_on_batch(latent_real, np.ones((desc_batch, 1)))
		d_loss_fake = self.discriminator.train_on_batch(latent_fake, np.zeros((desc_batch, 1)))
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


		# ---------------------
		#  Train Generator
		# ---------------------

		# Select a random batch of images
		idx = np.random.randint(0, Xbathy_train.shape[0], batch_size)

		bpatchs = Xbathy_train[idx]
		bpatchs_means = Xbathy_train_means[idx]


		# Train the generator

		g_loss = self.adversarial_autoencoder.train_on_batch([bpatchs, bpatchs_means], [bpatchs, bpatchs_means, np.ones((batch_size, 1))])

		# Plot the progress
		print ("bathy %d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))


		d_loss_hist.append(d_loss)
		g_loss_hist.append((g_loss[0],g_loss[1]))

		# If at save interval => save generated image samples
		if epoch % sample_interval == 0:
			self.sample_images(epoch)
			self.sample_autoencoder(epoch, bpatchs, bpatchs_means,"bathy_aae")

				# plotting the metrics
			plt.plot(d_loss_hist,linewidth=0.5)
			plt.plot(g_loss_hist,linewidth=0.5)
			plt.title('Model loss')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			plt.legend(['Dloss', 'Dacc','AEloss','AEmse'], loc='center right')
			plt.show()
			plt.savefig("metrics_aae_bathy/aae_bathy_metrics.png")
			plt.close()
	#save params once done training
	self.save_model()
	self.save_latentspace([Xbathy_train,Xbathy_train_means],"z_bathy")

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
aae = AdversarialAutoencoder()
aae.train(epochs=200000, batch_size=32, sample_interval=500)
