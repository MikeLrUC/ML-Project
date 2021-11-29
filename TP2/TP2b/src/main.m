% Test Networks: 
%   main(datasets, "test", ["netfile1", "netfile2"]);
%
% Train(+Test) (MLNN Network): 
%   main(datasets, 
%       "network", layrecnet",
%       "cnn", false,
%       "hidden", 10,
%       "fn", "traingd",
%       "delay", 2,
%       "encoding", [20],
%       "save", false)
%
% Train(+save) (MLNN Network): 
%   main(datasets, 
%       "network", "feedforwardnet",
%       "cnn", false,
%       "hidden", 10,
%       "fn", "traingd")

function main(datasets, varargin)
    % Default Arguments
    split = 0.7;
    save = true;
    seeds = 1;
    encoding = [];
    test = [];
    delays = [];

    while ~isempty(varargin)
        switch(lower(varargin{1}))
            case "hidden"
                hidden = varargin{2};
            case "fn"
                functions = varargin{2};
            case "encoding"
                encoding = varargin{2};
            case "save"
                save = varargin{2};
            case "split"
                split = varargin{2};
            case "seeds"
                seeds = varargin{2};
            case "network"
                networks = varargin{2};
            case "cnn"
                cnn = varargin{2};
            case "delay"
                delays = varargin{2};
            case "test"
                test = varargin{2};
            otherwise
                error(['Unexpected Argument: ' varargin{1}]);
        end
        varargin(1:2) = [];
    end
  

    for file = datasets
      % Load Dataset
      [P, T] = load_data(file);

      % Split Dataset
      [X_train, X_test, y_train, y_test] = train_test_split(P, T, "split", split, "encoding", encoding);
       
      % Train Networks
      if isempty(test)
          if cnn == true
              % Train Convolutional Neural Networks
              NN = cnn(X_train, y_train, networks, functions, hidden, delays);
          else
              % Train Multi Layer Neural Networks
              NN = mlnn(X_train, y_train, networks, functions, hidden, delays, "save", save, "seeds", seeds);
          end
      else
          NN = test;
      end
      
      % Test Networks && Gather Relevant Information
      evaluate(NN, X_test, y_test);
    end
end


function evaluate(N, X, T)

    for i = 1:size(N, 1)
        % Load network (from file path/cell array)
        if iscell(N(i))
            net = N{i, 1};
            name = N{i, 2};
        else
            S =  load(N(i));
            net = S.NN;
            name = N(i);
        end

        % Test the network
        disp("Testing Network: "+ name);
        Y = net(X, 'UseParallel','yes','UseGPU','yes');
    
        % Evaluate Acording to performance metrics
        p = perform(net, Y, T);
        [c, cm, ~ , perf] = confusion(T, Y); 

        % Debug
        disp(" => Performance (MSE) :" + p);
        disp(" => Confusion Matrix"); disp(cm);
        disp(" => Accuracy: " + (1 - c));
        disp(" => Sensitivity: "); disp(perf(:, 3));
        disp(" => Specificity: "); disp(perf(:, 4));
    end       
end

function [X_train, X_test, y_train, y_test] = train_test_split(P, T, varargin)
    % Default Parameters
    split = 0.7; % (70% training / 30% testing)
    encoding = []; % no encoding
    
    % Load Optional Arguments
    while ~isempty(varargin)
        switch(lower(varargin{1}))
            case 'encoding'
                encoding = varargin{2};
            case 'split'
                split = varargin{2};
            otherwise
                error(['Unexpected Argument: ' varargin{1}]);
        end
        varargin(1:2) = [];
    end

    % Split Index
    s = floor(length(P) * split);

    % Setup Training Data
    X_train = P(:, 1:s);
    y_train = T(:, 1:s);
  
    % "sub-sampling" (pos/inter)-ictal periods (normalize dataset)
    A = find(y_train(1, :) == 1); % pos-ictal + inter-ictal
    B = find(y_train(2, :) == 1 | y_train(3, :) == 1); % pre-ictal + ictal

    S = sort([A(:, randi(length(A), 1, length(B))), B]); % sample
    X_train = X_train(:, S);
    y_train = y_train(:, S);
   
    % Setup Test Data
    X_test = P(:, s+1:end);
    y_test = T(:, s+1:end);
    
    % Use Autoencoder
    [X_train, components] = stacked_encoder(X_train, encoding);
    for encoder = components
        X_test = encode(encoder{1}, X_test);
    end   
end


function  [P, T] = load_data(file) 
    % Load Dataset
    disp("Loading Dataset " + file);
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