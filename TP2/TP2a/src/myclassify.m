function [arg] = myclassify(P, filled_indexes, architecture, transfer_function)
    % Setting Seed    
    rng(0);
    
    % Setting up Input (P)
    len = ceil(length(P)/10);
    P = P(:,filled_indexes);
    
    % Loading Prototype Inputs
    dataset = load('P_500.mat');
    dataset = dataset.P(:, filled_indexes);
    
    % Setting up Classifier Target
    identity = repmat(eye(10), 1, len);
    identity = identity(:, filled_indexes);
    
    if (architecture == "Filter + Classifier (1 Layer)")
        % -- Loading -- %
        
        % Loading Filter Target (T)
        T = load('PerfectArial.mat');
        T = repmat(T.Perfect, 1, len);
        T = T(:, filled_indexes);
        
        % -- Filter: Associative Memory -- %
      
        % Weights Evaluation for prototypes
        W = T * pinv(dataset);

        % Filter Output 
        P = W * P;             % Filtered P
        dataset = W * dataset; % Filtered dataset
    end
    
    if (architecture == "Classifier (2 Layers)")
        hidden_n = 100;
        % -- Classifier 2 Layers -- %
        
        % Configuration: Shallow Network w/ 1 Input, 2 Layers, With bias,
        %   - Input connects to only Layer
        %   - No Layer connects to other Layer
        %   - Only Layer connects to Output
        
        net = network(1,2,[1;1],[1;0], [0 0; 1 0],[0,1]);
        
        % Setting up: Input size and Layer Nodes
        net.inputs{1}.size = 256;
        net.layers{1}.size = hidden_n; % Hidden Layer
        net.layers{2}.size = 10;
        
        % Setting up: Random Weights and Biases
        random_weights = rand(hidden_n, 256);
        random_biases = rand(hidden_n,1);

        net.IW{1,1} = random_weights;
        net.b{1,1} = random_biases;
        
        random_weights = rand(10, hidden_n);
        random_biases = rand(10,1);

        net.LW{2,1} = random_weights;
        net.b{2,1} = random_biases;
        
        % Setting up: Transfer Function (Activation Function) of last Layer
        if (transfer_function == "Linear")
            net.layers{2}.transferFcn = "purelin";
        elseif (transfer_function == "Sigmoidal")
            net.layers{2}.transferFcn = "logsig";
        elseif (transfer_function == "Hardlim")
            net.trainFcn = "trainc";
            net.adaptFcn = "learnp";
            net.layers{2}.transferFcn = "hardlim";
        end
    else
        % -- Classifier 1 Layer -- %

        % Configuration: Shallow Network w/ 1 Input, 1 Layer, With bias,
        %   - Input connects to First Layer (hidden)
        %   - 1st Layer connects to 2nd Layer
        %   - 2nd Layer connects to Output
        
        net = network(1, 1, 1, 1, 0, 1);
    
        % Setting up: Input size and Layer Nodes
        net.inputs{1}.size = 256;
        net.layers{1}.size = 10;

        % Setting up: Random Weights and Biases

        random_weights = rand(10, 256);
        random_biases = rand(10,1);

        net.IW{1,1} = random_weights;
        net.b{1,1} = random_biases;
    end

    % Setting up: Dataset Split (train, validation, test)
    net.divideFcn = 'dividerand'; % Random Split
    
    % Setting up: Training Method
    net.trainFcn = "traingd"; % Gradient Descend
    
    % Setting up: Transfer Function (Activation Function) of First Layer
    if (transfer_function == "Linear")
        net.layers{1}.transferFcn = "purelin";
    elseif (transfer_function == "Sigmoidal")
        net.layers{1}.transferFcn = "logsig";
    elseif (transfer_function == "Hardlim")
        net.trainFcn = "trainc";
        net.adaptFcn = "learnp";
        net.layers{1}.transferFcn = "hardlim";
    end
    
    % Extra Network Training Parameters
    net.performParam.lr = 30;
    net.trainParam.epochs = 20000;
    net.performFcn = 'mse';

    % Training Neural Network (Input is Weights * dataset and Target is identity)
    [net, tresults] = train(net, dataset, identity);
    
    % Displaying Training Results
    figure(); plotperform(tresults)
 
    
    % Classifing P2
    Y = net(P);
    disp(Y)
    [~, arg] = max(Y);
    disp(arg)
    return
end