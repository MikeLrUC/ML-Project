function accuracy(n_exp, dataN)
    disp("   Rede Neuronal" + repelem(' ', 10) + "Train Accuracy" + repelem(' ', 10) + "Test Accuracy")
    
    for classifier = ["Filter + Classifier (1 Layer)", "Classifier (1 Layer)", "Classifier (2 Layers)"]
        for transferFcn = ["Linear","Hardlim","Sigmoidal"]
            for seed = 0:n_exp - 1
                    P_train = load("P_" + dataN + ".mat");
                    P_train = P_train.P;
                    T_train = repmat(eye(10), 1, dataN/10);

                    P_test = load("AccuracyTest.mat");
                    P_test = P_test.P;
                    T_test = repmat(eye(10), 1, 5);
                if (classifier == "Filter + Classifier (1 Layer)")
                    name = "FC1_" + transferFcn + "_" + dataN + "_" + seed;
                            
                        % Loading Filter Target (T)
                        Perfect = load('PerfectArial.mat');
                        Perfect_target = repmat(Perfect.Perfect, 1, dataN/10);
                        
                        % -- Filter: Associative Memory -- %

                        % Weights Evaluation for prototypes
                        W = Perfect_target * pinv(P_train);
                        
                        % Filter Output 
                        P_train = W * P_train; % Filtered P_train
                        P_test = W * P_test; % Filtered P_test
                    
                elseif (classifier == "Classifier (1 Layer)")
                    name = "C1_" + transferFcn + "_" + dataN + "_" + seed;
                elseif (classifier == "Classifier (2 Layers)")
                    name = "C2_" + transferFcn + "_" + dataN + "_" + seed;
                end
                net = load(name + ".mat");
                net = net.net;
                
                % Train Accuracy
                Y_train = net(P_train);
                [~, arg_train] = max(Y_train);
                [~, exp_arg_train] = max(T_train);
                train_acc = "" + sum(arg_train == exp_arg_train) / size(P_train, 2);
                
                % Test Accuracy
                Y_test = net(P_test);
                [~, arg_test] = max(Y_test);
                [~, exp_arg_test] = max(T_test);
                test_acc = "" + sum(arg_test == exp_arg_test) / size(P_test, 2);
                
                % Display
                disp(name + repelem(' ', 30 - strlength(name)) + train_acc + repelem(' ', 25 - strlength(train_acc)) + test_acc )
            end
        end
    end
end