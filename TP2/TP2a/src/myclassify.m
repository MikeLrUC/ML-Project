function [arg] = myclassify(P, filled_indexes, architecture, transfer_function)
   
    % Setting up Input (P)
    P = P(:,filled_indexes);
    
    % Loading Prototype Inputs
    dataset = load("P_1000.mat");
    dataset = dataset.P;
    Q = size(dataset, 2);
    
    if (architecture == "Filter + Classifier (1 Layer)")
        % -- Loading -- %
        
        % Loading Filter Target (T)
        T = load('PerfectArial.mat');
        T = repmat(T.Perfect, 1, Q/10);
        
        % -- Filter: Associative Memory -- %
      
        % Weights Evaluation for prototypes
        W = T * pinv(dataset);

        % Filter Output 
        P = W * P; % Filtered P
        
        if (transfer_function == "Linear")
           net_name = "FC1_Linear_1000_0.mat";
        elseif (transfer_function == "Sigmoidal")
           net_name = "FC1_Sigmoidal_1000_0.mat";
        elseif (transfer_function == "Hardlim")
           net_name = "FC1_Hardlim_1000_0.mat";
        end
    
    elseif (architecture == "Classifier (1 Layer)")
        
        if (transfer_function == "Linear")
           net_name = "C1_Linear_1000_0.mat";
        elseif (transfer_function == "Sigmoidal")
           net_name = "C1_Sigmoidal_1000_0.mat";
        elseif (transfer_function == "Hardlim")
           net_name = "C1_Hardlim_1000_0.mat";
        end
  
    elseif (architecture == "Classifier (2 Layers)")
        if (transfer_function == "Linear")
           net_name = "C2_Linear_1000_0.mat";
        elseif (transfer_function == "Sigmoidal")
           net_name = "C2_Sigmoidal_1000_0.mat";
        elseif (transfer_function == "Hardlim")
           net_name = "C2_Hardlim_1000_0.mat";
        end
    end
    
    net = load(net_name);
    net = net.net;
  
    % Classifing P2
    Y = net(P);
    disp(Y)
    [~, arg] = max(Y);
    disp(arg)
    return
end