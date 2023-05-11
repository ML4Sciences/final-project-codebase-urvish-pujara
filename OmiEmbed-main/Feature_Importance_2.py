import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# # Generate example data
# X, _ = make_blobs(n_samples=1000, centers=2, random_state=42)
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define and train VAE model
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Lambda
# from tensorflow.keras.losses import mse
# from tensorflow.keras import backend as K

# VAE architecture
input_dim = X.shape[1]
latent_dim = 2

# Encoder
inputs = Input(shape=(input_dim,))
h1 = Dense(10, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h1)
z_log_var = Dense(latent_dim)(h1)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
h2 = Dense(10, activation='relu')(z)
outputs = Dense(input_dim, activation='linear')(h2)

# Define VAE model
vae = Model(inputs=inputs, outputs=outputs)

# VAE loss
reconstruction_loss = mse(inputs, outputs)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = reconstruction_loss + kl_loss
vae.add_loss(vae_loss)

# Compile VAE model
vae.compile(optimizer='adam')
vae.summary()

# Train VAE model
vae.fit(X_train, X_train, epochs=100, batch_size=32, validation_data=(X_test, X_test))

# Extract latent representations
encoder = Model(inputs=inputs, outputs=z_mean)  # Use z_mean as the latent representation
latent_representations = encoder.predict(X_train)

# Calculate feature importance
lin_reg = LinearRegression()
feature_importance = []

for i in range(latent_representations.shape[1]):
    # Fit linear regression model with latent representation as input and original feature as output
    lin_reg.fit(latent_representations[:, i].reshape(-1, 1), X_train[:, i].reshape(-1, 1))
    # Extract feature importance (slope of the linear regression model)
    feature_importance.append(np.abs(lin_reg.coef_[0][0]))

# Sort features based on feature importance
sorted_features = np.argsort(feature_importance)

# Print feature importance
print("Feature Importance (Method 1):")
for i in range(len(sorted_features)):
    print(f"Feature {sorted_features[i]}: Importance = {feature_importance[sorted_features[i]]:.4f}")
