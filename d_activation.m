function y = d_activation(x)
% Hyperbolic
y = (4*exp(2*x))./((exp(2*x) + 1).^2);

% Logistic
%y = exp(-x)./((exp(-x) + 1).^2);
