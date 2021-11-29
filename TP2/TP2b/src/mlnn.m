function [NN] = mlnn(X, T, network, fn, hidden, delay)
    % Convert features to cell arrays
    X_train = num2cell(X, 1);
    y_train = num2cell(T, 1);

    % Debug
    disp(" => Network: " + network);
    disp(" => TrainFn: " + fn);
    disp(" => HiddenSize: "+ hidden);
    if ~isempty(delay)
        disp(" => Delay: "+ delay);
    end
    
    % Train Neural Network
    if network == "layrecnet"
       net = layrecnet(1:delay, hidden, fn);
       [Xs, Xi, Ai, Ts] = preparets(net, X_train, y_train);
       NN = train(net, Xs, Ts, Xi, Ai, 'UseParallel','yes','UseGPU','yes'); 
    else
       net = feedforwardnet(hidden, fn);
       NN = train(net, X_train, y_train, 'UseParallel','yes','UseGPU','yes');  
    end
end
