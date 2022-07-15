function y = activation(x)
% Hyperbolic
y = (1 - exp(-2*x))./(1 + exp(-2*x));

% Logistic
%y = 1./(1 + exp(-x));