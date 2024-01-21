function F = CNN_Descriptor(img)
    % Load a pre-trained CNN (here, AlexNet)
    net = alexnet;

    % Read and preprocess the image
    img = imresize(img, net.Layers(1).InputSize(1:2)); % Resize image to match network's expected input size

    % Extract features (activations from a specific layer)
    % For AlexNet, 'fc7' is a commonly used layer for feature extraction
    cnn_features = activations(net, img, 'fc7');

    % Flatten the features to get a feature vector
    F = cnn_features(:)';
end