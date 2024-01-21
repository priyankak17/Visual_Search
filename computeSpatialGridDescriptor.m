function F = computeSpatialGridDescriptor(img, num_rows, num_cols)
    % Calculate cell size based on the image size and the number of rows and columns
    [height, width, ~] = size(img);
    cell_height = floor(height / num_rows);
    cell_width = floor(width / num_cols);

    F = zeros(num_rows * num_cols, 3); % Initialize the descriptor

    for row = 1:num_rows
        for col = 1:num_cols
            % Define the coordinates for the current cell
            y1 = (row - 1) * cell_height + 1;
            y2 = row * cell_height;
            x1 = (col - 1) * cell_width + 1;
            x2 = col * cell_width;

            % Extract the current cell from the image
            cell = img(y1:y2, x1:x2, :);

            % imshow(cell);

            % Compute the average color for the current cell
            avg_color = mean(mean(cell, 1), 2);

            % Store the average color in the descriptor
            index = (row - 1) * num_cols + col;
            F(index, :) = avg_color;
        end
    end

    % Reshape the descriptor into a vector for further processing if needed
    F = reshape(F, 1, []);
end
