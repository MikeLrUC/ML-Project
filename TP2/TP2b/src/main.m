
function main(varargin)
    for i = 1:nargin
      file = varargin{i};
      [P, T] = load_data(file);
      [X_train, X_test, y_train, y_test] = train_test_split(P, T);
      mlnn(X_train, y_train, "network", "feedforwardnet")
    end
end


function [X_train, X_test, y_train, y_test] = train_test_split(P, T, varargin)
    % Default value
    if nargin == 1
       split = varargin{1};
    else
       split = 0.7; % (70% training / 30% testing)
    end
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