load SUBMARINE.MAT

N_antennas = size(X,1); % Number of antennas
snapshots = size(X,2); % Number of snapshots

% DOA estimation
angles = -90:90; % Angle range for DoA estimation
num_sources = 4;

P = zeros(size(angles));
% Autocorrelation Matrix
Rx = X * X' / size(X, 2);
% Eigenvalue Decomposition
[E, D] = eig(Rx);
% sort the eigen value
[~, index] = sort(diag(D), 'descend');
% get the noise subspace
En = E(:, index(num_sources+1:end));
for i = 1:length(angles)
    A = array_manifold(angles(i), N_antennas, d);
    P(i) = 1 / (A' * En * En' * A);
end
music_spectrum = 10 * log10(P / max(P)); % convert to dB 

angles = 0:180;
plot(angles, music_spectrum);
xlabel('Angle (degrees)');
ylabel('Spatial Spectrum (dB)');

% function to get array manifold matrix: A
function A = array_manifold(doa, N, d) % N is the number of antennas
    doa_radians = deg2rad(doa);
    array_index = 0:N-1;
    A = exp(1j * 2 * pi * d * array_index.' * sin(doa_radians));
end