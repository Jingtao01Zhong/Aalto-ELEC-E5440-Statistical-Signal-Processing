clear all
%% Generate the input and the output data
N = 100;
a = 5;
b = 2.8;

% Input data
x = 1 + (2.5 - 1) * rand(N, 1);

% Signal Model
y_true = a * x.^b;

% Add Gaussian Noise, variance == 1
v = randn(N,1);
y_noisy = y_true + v;

% replace 10% of the noisy measurements by outliers
num_outliers = 0.1 * N;
outlier_index = randperm(N, num_outliers);
y_noisy(outlier_index) = 32 * x(outlier_index);

% re-order the x and y for figure
[x, order] = sort(x);
y_noisy = y_noisy(order);
y_true = y_true(order);

%% Least Squares Estimator
% turn the Exponential function to Linear function: y_ln = a_ln + b*x_ln;
y_ln = log(y_noisy);
x_ln = log(x);

% observation matrix: H
H = [ones(N,1), x_ln];
% Estimator: Theta
Theta = (H.' * H)^-1 * H.' * y_ln;
a_Theta = exp(Theta(1));
b_Theta = Theta(2);

%% M-Estimatior
% define the target Converges
epsilon = 1e-10;

% first estimation: LS estimation
y_ls = a_Theta * x.^b_Theta;
% residual errors
res = y_noisy - y_ls;

alpha = 5;
psi = zeros(1,N);
omega = zeros(1,N);
diff = Inf;
num_iterations = 0;
a_converge = a_Theta;
b_converge = b_Theta;
while diff > epsilon
    % Andrew’s weighting function
    for i = 1 : N
        if(abs(res(i)) <= alpha*pi)
                psi(i) = sin(res(i)/alpha); %variance == 1
                omega(i) = psi(i) / res(i);
        end
    end
    W =  diag(omega);
    
    % Estimate a and b
    Theta_M = (H.' * W * H)^-1 * H.' * W * y_ln;
    a_Theta_M = exp(Theta_M(1));
    b_Theta_M = Theta_M(2);
    
    % calculate the sum of residuals
    y_M = a_Theta_M * x.^b_Theta_M;
    res_M = y_noisy - y_M;
    diff = sum(res_M - res);
    res = res_M;
    num_iterations = num_iterations +1;
    a_converge = [a_converge a_Theta_M];
    b_converge = [b_converge b_Theta_M];
end
y_M = a_Theta_M .* x.^b_Theta_M;
%% Plot the result
figure(1)
hold on
plot(x,y_true,'--', 'Linewidth', 2);
plot(x,y_noisy,'o')
plot(x,y_ls,'m')
plot(x,y_M,'g')
legend(["True Model", "Noisy Model", "Least Square", "M-estimation"], 'Location', 'best')
xlabel("X")
ylabel("Y")
hold off

figure(2)
hold on
plot(0:num_iterations,a_converge,'-o');
plot(0:num_iterations,b_converge,'-o');
plot(0:num_iterations,a.*ones(1,num_iterations+1),"--");
plot(0:num_iterations,b.*ones(1,num_iterations+1),"--");
legend(["converge a", "converge b", "real a", "real b"], 'Location', 'best')
xlabel("num of iterations")
ylabel("convergent and real parameter")
hold off

t = -alpha*pi : 0.01 : alpha*pi;
Andrew_W_func = sin(t/alpha);
figure(3)
plot(t,Andrew_W_func)
title("Andrew’s weighting function")

figure(4)
subplot(2,1,1)
plot(1:N,abs(res))
subtitle("absolute value of the residual error vs the sample index n")
subplot(2,1,2)
plot(1:N,omega)
subtitle("weights corresponding to the signal samples after convergence")