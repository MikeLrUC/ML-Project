function accuracy = my_fuzzy(root, fis_name, X_train, y_train, X_test, y_test)
    fis = readfis(fullfile(root, "fuzzy-systems", fis_name + ".fis"));
    [inputs, outputs, rules] = getTunableSettings(fis);
    
    opts = tunefisOptions();
    opts.ValidationWindowSize = 2;
    opts.ValidationTolerance = 0.05;
    opts.MethodOptions.MaxGenerations = 10;
    opts.NumMaxRules = 25;
    opts.OptimizationType = 'learning';
    
    tuned_fis = tunefis(fis, [inputs; []; rules], X_train, y_train, opts);
    
    output = evalfis(tuned_fis, X_test);
    accuracy = sum(round(output) == y_test(:)) / length(y_test);
    disp("Accuracy: " + accuracy)
    fuzzy(tuned_fis)
end