root = "data";
ratio = 0.7;
f = fopen("results.txt", "w");
for n_rules = [9, 25]
    for method = ["grayscale_average", "rgb_average"]
       for fis_name = ["mamdani", "sugeno"]
           fprintf(f, "### N_RULES: " + n_rules + " METHOD: " + method + " FIS: " + fis_name + " ###\n");
           [total_acc, class_accs] = main(root, method, ratio, fis_name, n_rules);
           fprintf(f,"Total Accuracy: " + total_acc + "\n");
           for i = 1 : 6
                fprintf(f,"Class " + i + " Accuracy: " + class_accs(i) + "\n");
           end
       end
    end
end
fclose(f);
