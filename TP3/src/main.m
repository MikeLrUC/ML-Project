function main()
    root = "data";
    processed = "processed";
    method = "rgb_average";
    ratio = 0.7;
    try
        disp("Trying to Load Processed Data")
        % Load Saved Processed Data
        X_train = load(fullfile(root, processed, "X_train"));
        y_train = load(fullfile(root, processed, "y_train"));
        X_test = load(fullfile(root, processed, "X_test"));
        y_test = load(fullfile(root, processed, "y_test"));
    catch
        disp("Not Found: Processing data...")
        % Load and Transform Data
        [data, labels] = preprocess(root, method);

        % Split Dataset (Train/Test)
        [X_train, y_train, X_test, y_test] = split_dataset(data, labels, ratio);
        
        if ~exist(fullfile(root, processed), "dir")
            mkdir(fullfile(root, processed));
        end

        % Save Network
        save(fullfile(root, processed, "X_train"), "X_train");
        save(fullfile(root, processed, "y_train"), "y_train");
        save(fullfile(root, processed, "X_test"), "X_test");
        save(fullfile(root, processed, "y_test"), "y_test");
    end
    disp("Done!")
    
    % Fuzzy Inference System
    disp("Model")
end


function [data, labels] = preprocess(root, method)
    % Load Data and Generate Labels
    data = cell(1, 6);
    labels = cell(1, 6);
    
    for label = 1 : 6
        files = dir(fullfile(root, string(label), "*jpg"));
        n_images = length(files);
        data{label} = cell(1, n_images);
        labels{label} = repmat(label, 1, n_images);
        for n = 1 : n_images
            [~, data{label}{n}] = createMask(imread(files(n).name));
        end
    end
    
    % Transform Data
    data = transform(data, method);
end

function [X_train, y_train, X_test, y_test] = split_dataset(data, labels, ratio)
    % Train Cell Arrays   
    X_train = cell(1, 6);
    y_train = cell(1, 6);
    
    % Test Cell Arrays
    X_test = cell(1, 6);
    y_test = cell(1, 6);
    

    for label = 1 : 6
        % Dataset Length
        n = length(labels{label});
        
        % Split Index
        index = round(n * ratio);
        
        % Splitting
        X_train{label} = data{label}(:, 1 : index);
        y_train{label} = labels{label}(:, 1 : index);
        
        X_test{label} = data{label}(:, index + 1 : end);
        y_test{label} = labels{label}(:, index + 1 : end);
    end
    
    % Joining 
    X_train = horzcat(X_train{:});
    y_train = horzcat(y_train{:});
    
    X_test = horzcat(X_test{:});
    y_test = horzcat(y_test{:});
end
