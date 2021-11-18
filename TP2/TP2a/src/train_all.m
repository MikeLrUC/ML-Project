function [] = train_all(n_exp)
    for dataN = [200, 500, 1000]
        for classifier = ["Filter + Classifier (1 Layer)", "Classifier (1 Layer)", "Classifier (2 Layers)"]
            for transferFcn = ["Linear","Hardlim","Sigmoidal"]
                for seed = 0:n_exp - 1
                    disp("Saving: " + dataN + " | " + classifier + " | " + transferFcn + " | " + seed)
                    nn_train(dataN, classifier, transferFcn, seed, false);
                end
            end
        end
    end
end