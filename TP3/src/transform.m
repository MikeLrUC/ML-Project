function new_data = transform(data, method)
    new_data = cell(1, 6);
    
    if method == "rgb_average"
        n_features = 3;
    else
        n_features = 1;
    end
    
    % Looping Through Images
    for label = 1 : 6
        n_images = length(data{label});
        new_data{label} = zeros(n_features, n_images);
        for n = 1 : n_images
            img = data{label}{n};
            new_data{label}(:, n) = eval(method + "(img)");
        end
    end
end

function values = rgb_average(image)
    % Components
    r = image(:, :, 1);
    g = image(:, :, 2);
    b = image(:, :, 3);
    
    % Components Mean (not counting background, which is black (0))
    r = mean(r(r ~= 0)) / 255;
    g = mean(g(g ~= 0)) / 255;
    b = mean(b(b ~= 0)) / 255;
    
    values = [r; g; b];
end


function values = grayscale_average(image)
    values = rgb_average(image);
    values = 0.2989 * values(1) + 0.5870 * values(2) + 0.1140 * values(3);
end