function accuracy = my_fuzzy(root, fis_name, n_rules, X_train, y_train, X_test, y_test)
    fis = readfis(fullfile(root, "fuzzy-systems", fis_name + ".fis"));
    [inputs, ~, rules] = getTunableSettings(fis);
    
    opts = tunefisOptions();
    opts.UseParallel = true;
    opts.MethodOptions.MaxGenerations = 50;
    opts.NumMaxRules = n_rules;
    opts.OptimizationType = 'learning';
    
    tuned_fis = tunefis(fis, [inputs; []; rules], X_train, y_train, opts);
    
    output = evalfis(tuned_fis, X_test);
    accuracy = sum(round(output) == y_test(:)) / length(y_test);
    disp("Accuracy: " + accuracy)
    fuzzy(tuned_fis)
end