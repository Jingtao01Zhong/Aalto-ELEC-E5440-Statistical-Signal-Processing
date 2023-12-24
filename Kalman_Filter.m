T = 1; % sampling interval
N = 200;

noise_w = 9; % variance of the state noise
noise_v = 1; % variance of the measurement noise 

true_state = [0.5; 10]; % initial true state
estimate_state = [0.5; 11]; % initial state estimate

P = [noise_v noise_v/T; noise_v/T 2*noise_v/(T^2)]; %  initial state covariance
F = [1 T; 0 1]; % state matrix
G = [(T^2)/2 0; 0 1]; % state matrix (noise)
H = [1 0]; % observation matrix

Kalman_Gain_pos = zeros(1,N);
Kalman_Gain_vel = zeros(1,N);


position = zeros(1,N);
velocity = zeros(1,N);
pos_estimate = zeros(1,N);
vel_estimate = zeros(1,N);

pos_error_predict = zeros(1,N);
pos_error_estimate = zeros(1,N);
vel_error_predict = zeros(1,N);
vel_error_estimate = zeros(1,N);

for i = 1 : N

    % state equation
    noise_state = sqrt(noise_w) * randn(2,1);
    true_state = F * true_state + G * noise_state;
   
    % observation equation
    y = H * true_state + sqrt(noise_v) * randn;
    position(i) = true_state(1);
    velocity(i) = true_state(2);

    % Prediction Step
    predicted_state = F * estimate_state;
    P = F * P * F' + G*[noise_w, 0; 0, noise_w]*G'; % errror covariance
    pos_error_predict(i) = P(1,1);
    vel_error_predict(i) = P(2,2);

    % Correction Step
    kalman_gain = P * H' / (H * P * H' + noise_v); % Kalman Gain
    Kalman_Gain_pos(i) = kalman_gain(1);
    Kalman_Gain_vel(i) = kalman_gain(2);

    estimate_state = predicted_state + kalman_gain * (y - H * predicted_state);
    pos_estimate(i) = estimate_state(1);
    vel_estimate(i) = estimate_state(2);

    % Covariance of the correction step
    P = (eye(2) - kalman_gain * H) * P;
    pos_error_estimate(i) = P(1,1);
    vel_error_estimate(i) = P(2,2);

end

figure(1)
plot(position, velocity, 'b', pos_estimate, vel_estimate, 'r--');
xlabel('Position');
ylabel('Velocity');
legend('True State', 'Estimated State','Location', 'best');
title('Position and Velocity Tracking');

figure(2)
plot(1:N, pos_error_predict,'b', 1:N, pos_error_estimate, 'r');
xlabel('Sampling Time');
ylabel('Position Error Variance');
legend('Predicted  error variances', 'Estimated  error variances','Location', 'best');
title('Position Error Variance');

figure(3)
plot(1:N, vel_error_predict,'b', 1:N, vel_error_estimate, 'r');
xlabel('Sampling Time');
ylabel('Velocity Error Variance');
legend('Predicted  error variances', 'Estimated  error variances','Location', 'best');
title('Velocity Error Variance');

figure(4)
plot(1:N, Kalman_Gain_pos, 'b', 1:N, Kalman_Gain_vel, 'r');
xlabel('Sampling Time');
ylabel('Kalman Gain');
legend('Kalman Gain for Position', 'Kalman Gain for Velocity','Location', 'best');
title('Kalman Gains');