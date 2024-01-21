
%Global Colour Histogram
%Getting histValues for each channel
function F = ComputeRGBHistogram(img, Q) 

qimg= double(img);
qimg= floor(qimg.*Q);

bin = qimg(:,:,1)*Q^2 + qimg(:,:,2)*Q^1 + qimg(:,:,3);
vals=reshape(bin,1,size(bin,1)*size(bin,2));

%Plotting histValues for RBG together in one plot
F = hist(vals,Q^3);
F = F ./sum(F);



return;