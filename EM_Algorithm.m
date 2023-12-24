clear all
%% Generate the input and the output data
N = 64;
x1 = 64*rand(N,1);
x2 = 64*rand(N,1);
sigma = 0.1; % Variance of Noise is given
a1 = 8; b1 = 20;
a2 = -4; b2 = 10;
v1 = randn(N,1);
v2 = randn(N,1);

% Generate the data corresponding to the two models
y1 = a1 * x1 + b1 + v1;
y2 = a2 * x2 + b2 + v2;

% mix the data from two models randomly
outlier_index = randperm(2*N, 2*N);
y = [y1;y2];
x = [x1;x2];
y = y(outlier_index);
x = x(outlier_index);
[x, order] = sort(x);
y = y(order);

% Initial Estimation value
a1_est = 1; b1_est = 1;
a2_est = 0; b2_est = 0;

% Initialize the Weighted Quadratic Error Function
J1 = 1; J2 = 1;
i = 0;
%% EM Algorithm
while (J1 > 0.05 || J2 > 0.05)
    % E-step: Calculate the likelihood of each point belonging to each model
    r1 = (a1_est * x + b1_est - y); % Residuals for model 1
    r2 = (a2_est * x + b2_est - y); % Residuals for model 2
    % using log-likelihood function
    w1 = (r2.^2/sigma) ./ (r1.^2/sigma + r2.^2/sigma); 
    w2 = (r1.^2/sigma) ./ (r1.^2/sigma + r2.^2/sigma);
    w1 = w1 ./ sum(w1); % Normalize
    w2 = w2 ./ sum(w2);
    
    % Generating the weighting matrix
    W1 = diag(w1).' * diag(w1);
    W2 = diag(w2).' * diag(w2);

    % M-step: Update model parameters
    H = [ones(2*N,1) x];
    Theta1 = (H.' * W1 * H)^-1 * (H.' * W1 * y);
    b1_est = Theta1(1);
    a1_est = Theta1(2);

    Theta2 = (H.' * W2 * H)^-1 * (H.' * W2 * y);
    b2_est = Theta2(1);
    a2_est = Theta2(2);

    % Weighted Quadratic Error Function
    J1 = y.' * (W1 - W1 * H * (H.' * W1 * H)^-1 * H.' *W1) * y;
    J2 = y.' * (W2 - W2 * H * (H.' * W2 * H)^-1 * H.' *W2) * y;

    %% Plot the data
    i = i + 1;
    figure(i);
    hold on;
    scatter(x, y, 'b.');
    x_plot = 0:0.5:64;
    plot(x_plot, a1_est*x_plot + b1_est, 'r', 'LineWidth', 1);
    plot(x_plot, a2_est*x_plot + b2_est, 'g', 'LineWidth', 1);
    xlabel("x")
    ylabel("y")
    title("Times of Iterations:",i);
    legend(["data point", "Model 1", "Model 2"], 'Location', 'best')
    hold off
end