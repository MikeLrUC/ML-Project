function [SP_train, SS_train, A_train, SP_test, SS_test, A_test] ...
            = main(patient, network, encoding, train, class)

  % Default Parameters
  seed = 42;            % seed for training/testing network (easter-egg)
  fn = "traingd";       % Network Training Function;
  delay = 2;            % Delay (for layrecnet)
  hidden = 10;          % Hidden Layer Size
    
  % Clear Console
  clc;

  % Normalize Dataset
  [~, pname, ~] = fileparts(patient);
  if pname == "112502" % (35% training / 65% testing)
    split = 0.35; 
  else
    split = 0.7;       % (70% training / 30% testing)
  end

  % Debug
  disp("Global Parameters: ");
  disp(" => Train-Test split: " + (split * 100) + "/" + ((1 - split) * 100));
  disp(" => Seed: " + seed);
  disp(" => Encoding: " +  encoding);
  disp(" => Train: " + train);
  disp(" => Class: " + class);

  disp("Loading Dataset: " + patient);
  % Load Dataset
  [P, T] = load_data(patient);

  % Split && Encode Dataset
  [X_train, X_test, y_train, y_test] = train_test_split(pname, P, T, split, encoding, train);
   
  % Prepare data for CNN insertion (if it is a deep neural network)
  [X_train, X_test, y_train, y_test] = transform(network, X_train, X_test, y_train, y_test);

  % Network Name/ID
  if network == "layrecnet"
      ID = pname + "_" + regexprep(num2str(encoding), " +", "-") + "_" + fn + "_" + hidden + "_" + delay + "_" + seed;
  elseif network == "feedforwardnet"
      ID = pname + "_" + regexprep(num2str(encoding), " +", "-") + "_" + fn + "_" + hidden + "_" + seed;
  elseif network == "cnn"
      ID = pname + "_" + network + "_" + seed;
  elseif network == "lstm"
      ID = pname + "_" + regexprep(num2str(encoding), " +", "-") + "_" + network + "_" + seed;
  end
  
  % Set Seed
  rng(seed);
  
  % Train Networks
  if train == true

    % Debug
    disp("Training Network: "+ ID);

    if network == "cnn"
        % Train Convolutional Neural Networks
        NN = cnn(X_train, X_test, y_train, y_test, network);
    elseif network == "lstm"
        % Train Long Short Term Network
        if isempty(encoding)
            n_features = 29;
        else
            n_features = encoding(end);
        end
        NN = lstm_nn(n_features, X_train, X_test, y_train, y_test, network);
    else
        % Train Multi Layer Neural Networks
        NN = mlnn(X_train, y_train, network, fn, hidden, delay);
    end

    % Save Trained Neural Network

    % Create a directory for network storage
    root = fullfile("..", "data", "networks");
    if ~exist(root, 'dir')
        mkdir(root);
    end

    % Save Network
    save(fullfile(root, ID), "NN");

  else
      % Fetch Previously Trained Network
      NN = load(fullfile("..", "data", "networks", ID)).NN;
  end
  
  % Debug
  disp("Testing Network: "+ ID);

  % Gather Relevant Information (Train Data)
  [SP_train, SS_train, A_train] = evaluate(network, NN, X_train, y_train, class);

  % Debug
  disp(" => Train Data");
  disp("  => Accuracy: " + A_train);
  disp("  => Sensitivity: " + SS_train);
  disp("  => Specificity: " + SP_train);

  % Gather Relevant Information (Test Data)
  [SP_test, SS_test, A_test] = evaluate(network, NN, X_test, y_test, class);
  
  % Debug
  disp(" => Test Data");
  disp("  => Accuracy: " + A_test);
  disp("  => Sensitivity: " + SS_test);
  disp("  => Specificity: " + SP_test);
end


function [SP, SS, A] = evaluate(network, net, X, T, class)

    % Load Network (from path)
    if isstring(net)
        net = load(N, "NN");
    end

    % Test the network
    if network == "cnn"
        Y = classify(net, X);

        % Treat Data
        I = eye(3);
        Y = I(:, Y');  
        T = I(:, T');     
    elseif network == "lstm"
        Y = classify(net, X, 'MiniBatchSize', 27, 'SequenceLength','longest');
        
        % Treat Data
        I = eye(3);
        Y = I(:, Y');  
        T = I(:, T');  
    else
        Y = net(X, 'UseParallel','yes','UseGPU','yes');
    end

    % Evaluate Acording to performance metrics
    [c, cm, ~ , ~] = confusion(T, Y); 

    % Calculate Accuracy
    A = 1 - c;
     
    % Auxiliary Vector for matrix cell selection
    X = 1:3;
    X(class) = [];

    % Confusion Matrix (True/False - Positives/Negatives)
    TP = sum(cm(class, class), 'all');
    TN = sum(cm(X, X), 'all');
    FP = sum(cm(X, class), 'all');
    FN = sum(cm(class, X), 'all');

    % Calculate Sensitivity && Specificity
    SS = TP / (TP + FN);
    SP = TN / (TN + FP);
end       

function [X_train, X_test, y_train, y_test] = train_test_split(patient, P, T, split, encoding, train)
    % Split Index
    s = floor(length(P) * split);

    % Setup Training Data
    X_train = P(:, 1:s);
    y_train = T(:, 1:s);
  
    % "sub-sampling" (pos/inter)-ictal periods (normalize dataset)
    A = find(y_train(1, :) == 1);                      % pos-ictal + inter-ictal
    B = find(y_train(2, :) == 1 | y_train(3, :) == 1); % pre-ictal + ictal

    S = sort([A(:, randi(length(A), 1, length(B))), B]); % sample
    X_train = X_train(:, S);
    y_train = y_train(:, S);
   
    % Setup Test Data
    X_test = P(:, s+1:end);
    y_test = T(:, s+1:end);
       
    if isempty(encoding)
        return
    end

    % Auto Encoder ID
    ID = patient + "_" + regexprep(num2str(encoding), " +", "-");
    if train == true
        % Debug
        disp("Training Encoder...");

        % Train Auto Encoders
        [X_train, components] = stacked_encoder(X_train, encoding);
        for encoder = components
            X_test = encode(encoder{1}, X_test);
        end 

        % Create a directory for encoder storage
        root = fullfile("..", "data", "encoders");
        if ~exist(root, 'dir')
            mkdir(root);
        end

        % Save Components
        save(fullfile(root, ID), "components");
    else
        % Debug 
        disp("Loading Encoder...");

        % Use Previously trained autoencoder
        components = load(fullfile("..", "data", "encoders", ID)).components;
        for encoder = components  
            X_train = encode(encoder{1}, X_train);
            X_test = encode(encoder{1}, X_test);
        end  
    end
    % Debug
    disp("Encoding Completed!");
end

function [Xtr, Xtt, ytr, ytt] = transform(network, X_train, X_test, y_train, y_test)

    function [X_new, y_new] = blocks(X, y, gap, network)
        i = 1;
        X_new = [];
        y_new = [];
        while i <= length(X) - gap + 1
            dif = y(i : i + gap - 1) ~= y(i);
            if sum(dif) == 0                            % If the window has the same labels
                X_new = [X_new, X(:, i : i + gap - 1)];
                y_new = [y_new, y(i)];
                i = i + gap;
            else                                        % Otherwise
                [~, nexti] = max(dif);
                i = i + nexti - 1;
            end
        end
        
        if network == "cnn"
            % Blocks Reshaping (29x29x1xN_blocks) 
            X_new = reshape(X_new, [29, 29, 1, length(X_new) / gap]);
        elseif network == "lstm"
            n_blocks = floor(length(X_new) / gap);
            X_new = mat2cell(X_new', repmat(gap, n_blocks, 1));
            X_new = cellfun(@transpose, X_new, 'UniformOutput', false);
        end
        
        y_new = categorical(y_new');
        return
    end

    % Labels decoding
    [~, y_train] = max(y_train);
    [~, y_test] = max(y_test); 
    
    if network == "cnn"
        % Datapoints normalization
        min_val = min([X_train(:); X_test(:)]);
        max_val = max([X_train(:); X_test(:)]);
        
        X_train = (X_train - min_val) ./ max_val;
        X_test = (X_test - min_val) ./ max_val;
        
        % Block Building (29x29) 
        [Xtr, ytr] = blocks(X_train, y_train, 29, network);
        [Xtt, ytt] = blocks(X_test, y_test, 29, network);

    elseif network == "lstm"
        % Block Building (29x29) 
        [Xtr, ytr] = blocks(X_train, y_train, 29, network);
        [Xtt, ytt] = blocks(X_test, y_test, 29, network);
    else 
       Xtr = X_train;
       Xtt = X_test; 
       ytr = y_train;
       ytt = y_test;
    end
    return 
end

function  [P, T] = load_data(file) 
    % Load Dataset
    S = load(file);
    disp(S)

    % 1: "Pos/Inter-Ictal", 2:"Pré-Inctal", 3: "Ictal" classes
    % 1: [1 0 0], 2: [0 1 0], 3: [0 0 1] -> Class representation
    C = eye(3) ;

    % Seizure Vector (to be labeled)
    L = S.Trg';
    
    % Find Seizure Start (l) and End (h) points
    l = strfind(L, [0 1]) + 1;
    h = strfind(L, [1 0]);
    
    % Set Labels
    L(L == 0) = 1;
    for i = 1:length(l)
        L(:, max(l(i) - 901, 1):max(l(i) - 1, 1)) = 2;
        L(:, l(i):h(i)) = 3;
    end
    
    % Return Data
    P = S.FeatVectSel';  % Feature Matrix (features by row)
    T = C(:, L);         % Target (classified) Matrix
end