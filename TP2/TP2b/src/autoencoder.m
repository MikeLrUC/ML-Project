% Usage: autoencoder(X, 10)
function [features, autoencoder] = autoencoder(X, n_features)
    autoencoder = trainAutoencoder(X, n_features);
    features = encode(autoencoder, X);
    return
end