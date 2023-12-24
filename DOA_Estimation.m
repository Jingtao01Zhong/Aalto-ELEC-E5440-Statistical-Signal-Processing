N_antennas = 6; % Number of elements in the ULA
d = 0.5; % Element spacing (lambda/2)
snapshots = 512; % Number of snapshots
snr_db = [20 2]; % SNR values in dB
snr_scenarios = 10.^(snr_db/10);
doa_scenarios = [38, 93; 88, 93]; % DoA scenarios in degrees
doa_scenarios = doa_scenarios - 90;
num_experiments = 25; % Number of independent experiments

for i = 1:2
    % snr = 20 or snr = 2
    snr = snr_scenarios(i);
    for j = 1:2
        % DOA [38, 93] or [88, 93]
        doa = doa_scenarios(j,:);
        classical_spectrum_avg = zeros(1, 181);
        mvdr_spectrum_avg = zeros(1, 181);
        music_spectrum_avg = zeros(1, 181);

        for k = 1 : num_experiments
            % Generate 512*2 QPSK symbols
            qpsk_signal = qpsk_signal_generator(snapshots);
            A = array_manifold(doa, N_antennas, d);
            % assume the power of single symbol is 1, so the power of noise
            % is 1/sqrt(snr)
            X = A * qpsk_signal.' + randn(N_antennas, snapshots)*1/sqrt(snr);

            % DOA estimation
            angles = -90:90; % Angle range for DoA estimation
            classical_spectrum = classical_beamformer(X, d, N_antennas, angles);
            mvdr_spectrum = mvdr_beamformer(X, d, N_antennas, angles);
            music_spectrum = music_algorithm(X, d, N_antennas, angles, 2);

            % Accumulate results
            classical_spectrum_avg = classical_spectrum_avg + classical_spectrum;
            mvdr_spectrum_avg = mvdr_spectrum_avg + mvdr_spectrum;
            music_spectrum_avg = music_spectrum_avg + music_spectrum;
        end
        classical_spectrum_avg = classical_spectrum_avg / num_experiments;
        mvdr_spectrum_avg = mvdr_spectrum_avg / num_experiments;
        music_spectrum_avg = music_spectrum_avg / num_experiments;
        % Plot results (average over trials)
        angles = 0:180;
        figure;
        hold on;
        plot(angles, classical_spectrum_avg, 'b');
        plot(angles, mvdr_spectrum_avg, 'r');
        plot(angles, music_spectrum_avg, 'g');
        title(['DoA Estimation =' num2str(doa_scenarios(j,:)+90) ' with SNR = ' num2str(snr_db(i)) ' dB']);
        xlabel('Angle (degrees)');
        ylabel('Spatial Spectrum (dB)');
        legend('Classical', 'MVDR', 'MUSIC', 'Location', 'best');
        grid on;
    end
end

% function to generate 2 independent QPSK signals
function qpsk_signal = qpsk_signal_generator(N) % N is the number of symbols
    % generate 2 QPSK signals (2 DOA)
    qpsk_signal = (randi([0 1], N, 2) * 2 - 1) + 1j * (randi([0 1], N, 2) * 2 - 1);
    % The power of the symbol is 1
    qpsk_signal = qpsk_signal/sqrt(2);
end

% function to get array manifold matrix: A
function A = array_manifold(doa, N, d) % N is the number of antennas
    doa_radians = deg2rad(doa);
    array_index = 0:N-1;
    A = exp(1j * 2 * pi * d * array_index.' * sin(doa_radians));
end

% Classical beamformer
function P = classical_beamformer(X, d, N, angles)
    P = zeros(size(angles));
    for i = 1:length(angles)
        A = array_manifold(angles(i), N, d);
        P(i) = A' * X * X' * A;
    end
    P = 10 * log10(P / max(P)); % convert to dB
end

% MVDR beamformer
function P = mvdr_beamformer(X, d, N, angles)
    % Autocorrelation Matrix
    Rx = X * X' / size(X, 2);
    P = zeros(size(angles));
    for i = 1:length(angles)
        A = array_manifold(angles(i), N, d);
        P(i) = 1 / (A' / Rx * A);
    end
    P = 10 * log10(P / max(P)); % convert to dB
end

% MUSIC algorithm
function P = music_algorithm(X, d, N, angles, num_sources)
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
        A = array_manifold(angles(i), N, d);
        P(i) = 1 / (A' * En * En' * A);
    end
    P = 10 * log10(P / max(P)); % convert to dB
end
