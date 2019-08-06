function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
max_accuracy = 0; bestC = -1; bestsigma = -1;
accuracy = zeros(1, length(vals)*length(vals));
count = 1;
for C = 1:1:length(vals)
    for sigma = 1:1:length(vals)
        model = svmTrain(X, y, vals(C), @(x1, x2) gaussianKernel(x1, x2, vals(sigma)));
        predictions = svmPredict(model, Xval);
        tmp_accuracy = mean(double(predictions == yval));
        if(tmp_accuracy >= max_accuracy)
            max_accuracy = tmp_accuracy;            
            bestC = vals(C); bestsigma = vals(sigma);
        end;
    end;
end;

C = bestC; sigma = bestsigma;
% =========================================================================

end
