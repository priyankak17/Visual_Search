function descriptor = computeAverageColorDescriptor(img, num_rows, num_cols)
    % Get the size of the image
    [height, width, ~] = size(img);
    
    % Calculate the size of each grid cell
    cell_height = floor(height / num_rows);
    cell_width = floor(width / num_cols);

    % Initialize an array to store the descriptors
    descriptor = zeros(num_rows * num_cols, 3); % Assuming RGB color
    
    % Loop through each cell
    cell_index = 1;
    for row = 1:num_rows
        for col = 1:num_cols
            % Define the boundaries of the current cell
            row_start = (row - 1) * cell_height + 1;
            row_end = row * cell_height;
            col_start = (col - 1) * cell_width + 1;
            col_end = col * cell_width;

            % Extract the current cell from the image
            cell = img(row_start:row_end, col_start:col_end, :);

            % Calculate the average color in the cell
            avg_color = mean(mean(cell, 1), 2);

            % Store the average color in the descriptor
            descriptor(cell_index, :) = avg_color;

            cell_index = cell_index + 1;
        end
    end
end
