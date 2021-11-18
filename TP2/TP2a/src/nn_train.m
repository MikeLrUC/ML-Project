function [net] = nn_train(fileN, architecture, transfer_function, seed, plot)
    % Setting Seed    
    rng(seed);
    
    % Loading Prototype Inputs
    dataset = load("P_" + fileN + ".mat");
    dataset = dataset.P;
    Q = size(dataset, 2);
    
    % Setting up Classifier Target
    identity = repmat(eye(10), 1, Q/10);
    
    if (architecture == "Filter + Classifier (1 Layer)")
        % -- Loading -- %
        
        % Loading Filter Target (T)
        T = load('PerfectArial.mat');
        T = repmat(T.Perfect, 1, Q/10);
        
        % -- Filter: Associative Memory -- %
      
        % Weights Evaluation for prototypes
        W = T * pinv(dataset);

        % Filter Output 
        dataset = W * dataset; % Filtered dataset
    end
    
    if (architecture == "Classifier (2 Layers)")
        hidden_n = 50;
        
        % Neural Network
        net = network(1,2,[1;1],[1;0],[0 0; 1 0],[0,1]);
        
        % Setting up: Input size and Layer Nodes
        net.inputs{1}.size = 256;
        net.layers{1}.size = hidden_n; % Hidden Layer
        net.layers{2}.size = 10;
        
        % Setting up: Random Weights and Biases of First Layer
        random_weights = 0.1 * rand(hidden_n, 256);
        random_biases = 0.1 * rand(hidden_n,1);

        net.IW{1,1} = random_weights;
        net.b{1,1} = random_biases;
        
        % Setting up: Transfer Function (Activation Function) of First Layer
        if (transfer_function == "Linear")
            net.layers{1}.transferFcn = "purelin";
        elseif (transfer_function == "Sigmoidal")
            net.layers{1}.transferFcn = "logsig";
        elseif (transfer_function == "Hardlim")
            net.layers{1}.transferFcn = "hardlim";
        end
        
        % Setting up: Random Weights and Biases of Last Layer
        random_weights = 0.1 * rand(10, hidden_n);
        random_biases = 0.1 * rand(10,1);

        net.LW{2,1} = random_weights;
        net.b{2,1} = random_biases;
        
        % Setting up: Transfer Function (Activation Function) of Last Layer
        net.layers{2}.transferFcn = "hardlim";
        net.layerWeights{2,1}.learnFcn = 'learnp';
        net.biases{2}.learnFcn = 'learnp';
        
        % Setting up: Training Method
        net.trainFcn = "trainc";
        
        net.trainParam.epochs = 1000;
        
    else % -- Classifier 1 Layer -- %
        
        % Network
        net = network(1, 1, 1, 1, 0, 1);
    
        % Setting up: Input size and Layer Nodes
        net.inputs{1}.size = 256;
        net.layers{1}.size = 10;

        % Setting up: Random Weights and Biases

        random_weights = 0.1 * rand(10, 256);
        random_biases = 0.1 * rand(10,1);

        net.IW{1,1} = random_weights;
        net.b{1,1} = random_biases;
        
        % Setting up: Training Method & Epochs
        net.trainFcn = "traingd";
        net.trainParam.epochs = 20000;
        
        if (transfer_function == "Linear")
            net.layers{1}.transferFcn = "purelin";
        elseif (transfer_function == "Sigmoidal")
            net.layers{1}.transferFcn = "logsig";
        elseif (transfer_function == "Hardlim")
            net.layers{1}.transferFcn = "hardlim";
            
            net.inputWeights{1,1}.learnFcn = 'learnp';
            net.biases{1}.learnFcn = 'learnp';
            net.trainFcn = "trainc";
            net.trainParam.epochs = 1000;
        end
        
        
    end

    % Setting up: Dataset Split (train, validation, test)
    net.divideFcn = 'dividerand'; % Random Split
    
    % Extra Network Training Parameters
    net.performFcn = 'mse';
    
    % Training Neural Network (Input is Weights * dataset and Target is identity)
    [net, tresults] = train(net, dataset, identity);
    
    % Displaying Training Results
    if plot == true
        figure(); plotperform(tresults)
    end
    
        
    % Saving networks
    if (architecture == "Filter + Classifier (1 Layer)")
        name = "FC1_" + transfer_function + "_" + fileN + "_" + seed;
    elseif (architecture == "Classifier (1 Layer)")
        name = "C1_" + transfer_function + "_" + fileN + "_" + seed;
    elseif (architecture == "Classifier (2 Layers)")
        name = "C2_" + transfer_function + "_" + fileN + "_" + seed;
    end
    save(name, "net");
    
    return
end