% Usage Example: 
% X = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
% T = zeros(4)
% deepnet = stacked_encoder(X, T, [3,2])
function [features, components] = stacked_encoder(X, n_features_list)
    features = X;
    components = cell(1,length(n_features_list));
    for i = 1 : length(n_features_list)
       [features, autoenc] = autoencoder(features, n_features_list(i));
       components{i} = autoenc;
    end
    return
end