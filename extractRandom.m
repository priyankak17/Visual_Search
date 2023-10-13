function F=extractRandom(img)

F=rand(1,30);

% Returns a row [rand rand .... rand] representing an image descriptor
% computed from image 'img'

% Note img is a normalised RGB image i.e. colours range [0,1] not [0,255].


red = img(:,:,1);
red=reshape(red, 1, []);
average_red = mean(red);

green = img(:,:,1);
green = reshape(green, 1, []);
average_green = mean(green);

blue = img(:,:,1);
blue = reshape(blue, 1, []);
average_blue = mean(blue);


F = [average_red average_green average_blue];

return;
