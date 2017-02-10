function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

mse = 0;
for i = 1:rows(X)
    # regularization
    sqtheta = theta.^2;
    sqtheta(1) = 0;
    sumsqtheta = sum(sqtheta);
    regul = (lambda/(2*m))*sumsqtheta;
    
    #cost function:
    hoxi = sum((X(i,:)) .* (theta'));
    hoxi = sigmoid(hoxi);
    mse = mse + ((-y(i).*log(hoxi)) - (1-y(i)).*log(1-hoxi)) + regul;
    
    # gradient:
    j = 1;
    for valj=theta'
        if (j == 1)
            grad(j) = grad(j) + (hoxi - y(i)).*X(i,j);
        else
            grad(j) = grad(j) + (hoxi - y(i)).*X(i,j) + (lambda/m)*theta(j);
        endif
        j = j+1;
    endfor
    
endfor
J = (1/m)*mse;
grad = grad .* (1/m);




% =============================================================

end
