import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Lambda
# from tensorflow.keras.losses import mse
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate synthetic data
# X, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=0)
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

# Define VAE architecture
input_dim = X.shape[1]
latent_dim = 2

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(8, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = Input(shape=(latent_dim,))
h_decoded = Dense(8, activation='relu')(decoder_inputs)
outputs = Dense(input_dim, activation='linear')(h_decoded)

# Define VAE model
encoder = Model(inputs, z_mean)
decoder = Model(decoder_inputs, outputs)
vae = Model(inputs, decoder(z))
vae.compile(optimizer='adam', loss=mse)

# Train VAE
vae.fit(X_train, X_train, batch_size=32, epochs=100, verbose=0)

# Extract learned latent representation for input data
latent_representation = encoder.predict(X_test)

# Generate reconstructed data from latent representation
reconstructed_data = decoder.predict(latent_representation)

# Compute reconstruction error for each feature
reconstruction_error = np.mean(np.square(X_test - reconstructed_data), axis=0)

# Rank features by reconstruction error
feature_ranking = np.argsort(reconstruction_error)[::-1]

# Print feature importance ranking
print("Feature Importance Ranking:")
for i, feature_idx in enumerate(feature_ranking):
    print(f"{i + 1}. Feature {feature_idx}: Reconstruction Error = {reconstruction_error[feature_idx]:.4f}")

# print the most important feature
print(f"Most important feature: {feature_ranking[0]}")