function [NN] = cnn(X_train, X_test, y_train, y_test, network)
    % Debug
    disp(" => Network: " + network);

    % Train Neural Network
    if network == "cnn"
         layers = [
             imageInputLayer([29 29 1])
    
             convolution2dLayer(3,8,'Padding','same')
             batchNormalizationLayer
             reluLayer
    
             maxPooling2dLayer(2,'Stride',2)
    
             convolution2dLayer(3,16,'Padding','same')
             batchNormalizationLayer
             reluLayer
            
             maxPooling2dLayer(2,'Stride',2)
                
             convolution2dLayer(3,32,'Padding','same')
             batchNormalizationLayer
             reluLayer
            
             fullyConnectedLayer(3)
             softmaxLayer
             classificationLayer
         ];
       
         options = trainingOptions('sgdm', ...
                'InitialLearnRate' ,0.01, ...
                'MaxEpochs', 8, ...
                'Shuffle','every-epoch', ...
                'ValidationData', {X_test, y_test}, ...
                'ValidationFrequency',30, ...
                'Verbose',false, ...
                'Plots','training-progress');

         NN = trainNetwork(X_train, y_train, layers, options);
         % Agora para classificar faz-se classify(NN, X_test) e não NN(X_test)
    else
         
    end
    
end