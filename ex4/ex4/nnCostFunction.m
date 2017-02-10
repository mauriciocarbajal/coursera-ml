function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

#X = [ones(m, 1) X];

#bigDelta1 = zeros(hidden_layer_size,input_layer_size);  # 25x400
#bigDelta2 = zeros(num_labels,hidden_layer_size);        # 10x25

bigDelta1 = zeros(size(Theta1));
bigDelta2 = zeros(size(Theta2));

for i = 1:m
    # para el i-esimo training sample
    curr_x = X(i,:);
    curr_y = y(i);
    
    #layer 1 (400 units)
    a1 = curr_x;
    a1p1 = [1 a1]';
    
    #layer 2 (25 units)
    z2 = Theta1 * a1p1;
    a2 = sigmoid(z2);
    a2p1 = [1 a2']';
    
    #layer3 (10 units)
    z3 = Theta2 * a2p1;
    a3 = sigmoid(z3); #h(x)
    
    
    #vectorize correct solution y
    y_vec = zeros(1,num_labels);
    y_vec(curr_y) = 1;
    
    #cost J
    for k = 1:num_labels
        termino_i_k = (-(y_vec(k)) * log(a3(k))) - ((1 - (y_vec(k))) * log(1 - a3(k)));
        J = J + termino_i_k;
    endfor
    
    #Backpropagation and Gradient:
        #layer3
        delta3 = (a3 - y_vec');     #errors for layer3 (10 units)
        localgrad3 = (delta3 * a2p1');
        bigDelta2 = bigDelta2 + localgrad3;
    
        #layer2
        z2p1 = [1 z2']';# DON'T UNDERSTAND WHY I HAD TO DO THIS: z2's size is 25, not 26!
        delta2 = (Theta2' * delta3) .* sigmoidGradient(z2p1);
        delta2 = delta2(2:end);     #errors for layer2 (25 units)    
        localgrad2 = (delta2 * a1p1');
        bigDelta1 = bigDelta1 + localgrad2;
        
endfor

J = (1/m)*J;

#Regularization term added:
Theta1m = Theta1(:,2:end);
Theta2m = Theta2(:,2:end);
J = J + (lambda / (2*m)) * (sum(sum(Theta1m .^2)) + sum(sum(Theta2m .^2)));

#Regularization for gradient
regtermMatrix1 = (lambda / m) .* Theta1;
regtermMatrix2 = (lambda / m) .* Theta2;

regtermMatrix1(:,1) = 0;
regtermMatrix2(:,1) = 0;

Theta1_grad = (1/m) .* bigDelta1 + regtermMatrix1;
Theta2_grad = (1/m) .* bigDelta2 + regtermMatrix2;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%







% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
