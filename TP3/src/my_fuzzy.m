function  [acc, class_accs] = my_fuzzy(root, fis_name, n_rules, X_train, y_train, X_test, y_test)
    fis = readfis(fullfile(root, "fuzzy-systems", fis_name + ".fis"));
    [inputs, ~, rules] = getTunableSettings(fis);
    
    opts = tunefisOptions();
    opts.UseParallel = true;
    opts.MethodOptions.MaxGenerations = 50;
    opts.NumMaxRules = n_rules;
    opts.OptimizationType = 'learning';
    
    tuned_fis = tunefis(fis, [inputs; []; rules], X_train, y_train, opts);
    
    output = evalfis(tuned_fis, X_test);
    
    output = round(output);
    y_test = y_test(:);
    I = eye(6);
    [not_acc, cm, ~,~] = confusion(I(:, y_test), I(:, output));
    class_accs = diag(cm) ./ sum(cm, 2);
    acc = 1 - not_acc;
    disp("Total Accuracy: " + acc)
    for i = 1 : 6
       disp("Class " + i + " Accuracy: " + class_accs(i)) 
    end
end