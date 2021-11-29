function [MLNN] = mlnn(X, T, networks, functions, layers, delays, varargin)
    % Default Parameters
    seeds = 1;
    store = true;
    
    % Load Optional Arguments
    while ~isempty(varargin)
        switch(lower(varargin{1}))
            case 'seeds'
                seeds = varargin{2};
            case 'save'
                store = varargin{2};
            otherwise
                error(['Unexpected Argument: ' varargin{1}]);
        end
        varargin(1:2) = [];
    end

    % Prelocate space for the networks
    MLNN = cell(length(functions) * length(layers) * length(delays) * seeds, 2);

    % Train Networks
    i = 1;
    for s = 1:seeds    
        for n = networks
            for f = functions
                for h = layers
                    if n == "layrecnet"
                       for d = delays
                         [net, name] = mlnn_train(X, T, n, f, h, d, s, store); 
                         MLNN(i, :) = {net, name};
                         i = i + 1;
                       end
                    else
                         [net, name] = mlnn_train(X, T, n, f, h, [], s, store); 
                         MLNN(i, :) = {net, name};
                         i = i + 1;
                    end  
                end 
            end
        end    
    end
end


function [NN, ID] = mlnn_train(X, T, network, fn, hidden, delay, seed, store)
    % Set Seed
    rng(seed);

    % Convert features to cell arrays
    X_train = num2cell(X, 1);
    y_train = num2cell(T, 1);

    % Network Name/ID
    if network == "layrecnet"
        ID = "LRN_" + hidden + "_" + fn + "_" + delay + "_" + seed;
    else
        ID = "FFN_" + hidden + "_" + fn + "_" + seed;
    end

    % Debug
    disp("Training: " + ID);
    disp(" => Network: " + network);
    disp(" => TrainFn: " + fn);
    disp(" => HiddenSize: "+ hidden);

    if network == "layrecnet"
       disp(" => Delay: "+ delay);

       % Train Neural Network
       net = layrecnet(1:delay, hidden, fn);
       [Xs, Xi, Ai, Ts] = preparets(net, X_train, y_train);
       NN = train(net, Xs, Ts, Xi, Ai, 'UseParallel','yes','UseGPU','yes'); 
    else
       % Train Neural Network
       net = feedforwardnet(hidden, fn);
       NN = train(net, X_train, y_train, 'UseParallel','yes','UseGPU','yes');  
    end

    % Save Trained Neural Network
    if store == true
       % Create a directory for network storage
       root = fullfile("..", "data", "networks", network);
       if ~exist(root, 'dir')
           mkdir(root);
       end

       % Save Network
       save(fullfile(root, ID), "NN");
    end

end
