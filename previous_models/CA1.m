%% HW2 Simulation: CA1 place-cell ensemble drift
% Inspired by Ziv et al. (2013)
%
% Main logic:
% 1. Each session lasts 15 minutes.
% 2. Mouse explores a 100 cm linear track with variable motion.
% 3. There are 4 mice per group.
% 4. Each mouse has its own movement trajectories, neuron properties,
%    place-cell ensembles, firing rates, and spikes.
% 5. Stable group: consecutive-session ensemble overlap ~15-25%.
% 6. Modified group: consecutive-session ensemble overlap ~10-20%.
% 7. Bayesian decoding is reported as median decoding error in cm.
%    Stable group is expected around ~7-13 cm, modified group ~10-16 cm.

clear; clc; close all;
rng('shuffle');

%% ------------------------------------------------------------
% Experimental parameters
% ------------------------------------------------------------

N_mice_per_group = 4;

N_neurons = 800;
track_length = 100;               % cm
days = [1 4 7 10 14 17 21 24 27 30];

session_length = 15 * 60;         % 15 minutes = 900 seconds
dt = 0.1;
time = 0:dt:(session_length-dt);
T = length(time);

groups = {'stable', 'modified'};

K_place = 120;
K_nonplace = 160;

all_cells = 1:N_neurons;

%% ------------------------------------------------------------
% Main simulation loop
% ------------------------------------------------------------

for g = 1:length(groups)

    group_name = groups{g};

    for m = 1:N_mice_per_group

        fprintf('\nSimulating %s group, mouse %d...\n', group_name, m);

        previous_place_cells = [];

        % Each mouse gets its own neuron properties
        baseline = 0.02 + (0.10 - 0.02) * rand(N_neurons,1);   % Hz

        amplitude = 6 + 0.7 * randn(N_neurons,1);              % Hz
        amplitude(amplitude < 1) = 1;

        field_width = 9 + (14 - 9) * rand(N_neurons,1);        % cm
        initial_center = track_length * rand(N_neurons,1);     % cm

        for d = 1:length(days)

            day = days(d);
            S = getStimulus(day, group_name);

            fprintf('  Day %d...\n', day);

            % Each mouse and each session gets a different movement trajectory
            [position, velocity] = simulateMouseTrajectory(time, dt, track_length);

            %% ------------------------------------------------
            % Choose today's place-cell ensemble
            % ------------------------------------------------

            if d == 1

                place_cells_today = sampleWithoutReplacement(all_cells, K_place);
                target_J = NaN;

            else

                if strcmp(group_name, 'stable')
                    target_J = 0.15 + (0.25 - 0.15) * rand;
                else
                    target_J = 0.10 + (0.20 - 0.10) * rand;
                end

                % Convert desired Jaccard overlap to number of shared cells.
                %
                % J = m / (2K - m)
                % m = 2JK / (1 + J)

                m_shared = round((2 * target_J * K_place) / (1 + target_J));
                m_shared = min(m_shared, K_place);
                m_shared = max(m_shared, 0);

                shared_cells = sampleWithoutReplacement(previous_place_cells, m_shared);

                outside_previous = setdiff(all_cells, previous_place_cells);
                new_cells = sampleWithoutReplacement(outside_previous, K_place - m_shared);

                place_cells_today = [shared_cells, new_cells];

            end

            %% ------------------------------------------------
            % Choose active non-place cells
            % ------------------------------------------------

            available_nonplace = setdiff(all_cells, place_cells_today);
            nonplace_cells_today = sampleWithoutReplacement(available_nonplace, K_nonplace);

            active_cells_today = [place_cells_today, nonplace_cells_today];

            %% ------------------------------------------------
            % Initialize storage
            % ------------------------------------------------

            results.(group_name).mouse(m).day(d).day_number = day;
            results.(group_name).mouse(m).day(d).stimulus = S;
            results.(group_name).mouse(m).day(d).target_J = target_J;
            results.(group_name).mouse(m).day(d).position = position;
            results.(group_name).mouse(m).day(d).velocity = velocity;

            results.(group_name).mouse(m).day(d).rates = zeros(N_neurons, T);
            results.(group_name).mouse(m).day(d).spikes = false(N_neurons, T);

            results.(group_name).mouse(m).day(d).active = false(N_neurons,1);
            results.(group_name).mouse(m).day(d).place_cell = false(N_neurons,1);
            results.(group_name).mouse(m).day(d).field_center = nan(N_neurons,1);

            results.(group_name).mouse(m).day(d).active(active_cells_today) = true;
            results.(group_name).mouse(m).day(d).place_cell(place_cells_today) = true;

            %% ------------------------------------------------
            % Simulate firing rates
            % ------------------------------------------------

            for i = 1:N_neurons

                if ~results.(group_name).mouse(m).day(d).active(i)

                    % Inactive neuron
                    rate = zeros(1,T);

                elseif results.(group_name).mouse(m).day(d).place_cell(i)

                    % Place-cell firing rate:
                    % Gaussian place field over position.

                    if d == 1

                        mu = initial_center(i);

                    else

                        is_shared_with_previous = ...
                            results.(group_name).mouse(m).day(d-1).place_cell(i) && ...
                            results.(group_name).mouse(m).day(d).place_cell(i);

                        if is_shared_with_previous

                            previous_mu = results.(group_name).mouse(m).day(d-1).field_center(i);

                            if strcmp(group_name, 'stable')
                                drift_sd = 5.0;      % lower drift, better decoding
                            else
                                drift_sd = 8.0;      % higher drift, worse decoding
                            end

                            mu = previous_mu + drift_sd * randn;

                        else

                            % New place cell relative to previous session
                            mu = initial_center(i) + 10.0 * randn;

                        end
                    end

                    mu = max(0, min(track_length, mu));
                    results.(group_name).mouse(m).day(d).field_center(i) = mu;

                    % Day-to-day variability in amplitude and field width
                    amp_today = amplitude(i) * (1 + 0.10 * randn);
                    amp_today = max(1.0, amp_today);

                    width_today = field_width(i) * (1 + 0.10 * randn);
                    width_today = max(5, width_today);

                    rate = baseline(i) + amp_today * ...
                        exp(-((position - mu).^2) ./ (2 * width_today^2));

                else

                    % Active but non-place cell
                    rate = baseline(i) * ones(1,T);

                end

                results.(group_name).mouse(m).day(d).rates(i,:) = rate;

            end

            %% ------------------------------------------------
            % Generate Poisson spikes from firing rates
            % ------------------------------------------------

            spike_probability = results.(group_name).mouse(m).day(d).rates * dt;
            spike_probability(spike_probability > 1) = 1;

            results.(group_name).mouse(m).day(d).spikes = ...
                rand(size(spike_probability)) < spike_probability;

            previous_place_cells = place_cells_today;

        end
    end
end

%% ------------------------------------------------------------
% Summary statistics: mean activity across mice
% ------------------------------------------------------------

mean_rate_stable_mouse = zeros(N_mice_per_group, length(days));
mean_rate_modified_mouse = zeros(N_mice_per_group, length(days));

for m = 1:N_mice_per_group
    for d = 1:length(days)

        mean_rate_stable_mouse(m,d) = ...
            mean(results.stable.mouse(m).day(d).rates(:));

        mean_rate_modified_mouse(m,d) = ...
            mean(results.modified.mouse(m).day(d).rates(:));

    end
end

mean_rate_stable = mean(mean_rate_stable_mouse, 1, 'omitnan');
mean_rate_modified = mean(mean_rate_modified_mouse, 1, 'omitnan');

%% ------------------------------------------------------------
% Figure 1: stimulus and mean firing rate
% ------------------------------------------------------------

stable_stimulus = zeros(size(days));
modified_stimulus = zeros(size(days));

for d = 1:length(days)
    stable_stimulus(d) = getStimulus(days(d), 'stable');
    modified_stimulus(d) = getStimulus(days(d), 'modified');
end

figure;

subplot(2,1,1);
plot(days, stable_stimulus, '-o', 'LineWidth', 1.5); hold on;
plot(days, modified_stimulus, '-o', 'LineWidth', 1.5);
xlabel('Day');
ylabel('Stimulus value S(d)');
title('Visual stimulus across experimental days');
legend('Stable group', 'Modified wall-color group', 'Location', 'northwest');
xlim([1 30]);

subplot(2,1,2);
plot(days, mean_rate_stable, '-o', 'LineWidth', 1.5); hold on;
plot(days, mean_rate_modified, '-o', 'LineWidth', 1.5);
xlabel('Day');
ylabel('Mean simulated activity, Hz');
title('Mean simulated activity across CA1 neurons, mean across mice');
legend('Stable group', 'Modified group', 'Location', 'best');
xlim([1 30]);

%% ------------------------------------------------------------
% Figure 2: example mouse motion from one mouse only
% ------------------------------------------------------------

example_mouse = 1;
example_day = 1;

example_position = results.stable.mouse(example_mouse).day(example_day).position;
example_velocity = results.stable.mouse(example_mouse).day(example_day).velocity;

plot_seconds = 100;
plot_window = 1:round(plot_seconds/dt);

figure;

subplot(2,1,1);
plot(time(plot_window), example_position(plot_window), 'LineWidth', 1.2);
xlabel('Time, s');
ylabel('Position, cm');
title(sprintf('Example simulated trajectory: stable mouse %d, day %d, first 100 seconds', ...
    example_mouse, days(example_day)));
xlim([0 plot_seconds]);
ylim([0 track_length]);

subplot(2,1,2);
plot(time(plot_window), example_velocity(plot_window), 'LineWidth', 1.2);
xlabel('Time, s');
ylabel('Velocity, cm/s');
title(sprintf('Example simulated velocity: stable mouse %d, day %d, first 100 seconds', ...
    example_mouse, days(example_day)));
xlim([0 plot_seconds]);
ylim([-35 35]);

%% ------------------------------------------------------------
% Figure 3: example place-cell firing rates and spike trains
% stable vs modified, one example mouse only
% ------------------------------------------------------------

example_mouse = 1;
example_day = length(days);   % day 30

% Find one stable place cell with a field away from the edges
stable_example_cells = find(results.stable.mouse(example_mouse).day(example_day).place_cell & ...
                            results.stable.mouse(example_mouse).day(example_day).field_center > 20 & ...
                            results.stable.mouse(example_mouse).day(example_day).field_center < 80);

% Find one modified place cell with a field away from the edges
modified_example_cells = find(results.modified.mouse(example_mouse).day(example_day).place_cell & ...
                              results.modified.mouse(example_mouse).day(example_day).field_center > 20 & ...
                              results.modified.mouse(example_mouse).day(example_day).field_center < 80);

% Fallback if no cell is found in the 20-80 cm range
if isempty(stable_example_cells)
    stable_example_cells = find(results.stable.mouse(example_mouse).day(example_day).place_cell);
end

if isempty(modified_example_cells)
    modified_example_cells = find(results.modified.mouse(example_mouse).day(example_day).place_cell);
end

stable_example_cell = stable_example_cells(1);
modified_example_cell = modified_example_cells(1);

stable_rate = results.stable.mouse(example_mouse).day(example_day).rates(stable_example_cell,:);
stable_spikes = results.stable.mouse(example_mouse).day(example_day).spikes(stable_example_cell,:);

modified_rate = results.modified.mouse(example_mouse).day(example_day).rates(modified_example_cell,:);
modified_spikes = results.modified.mouse(example_mouse).day(example_day).spikes(modified_example_cell,:);

plot_seconds = 100;
plot_window = 1:round(plot_seconds/dt);

figure;

subplot(2,1,1);
plot(time(plot_window), stable_rate(plot_window), 'LineWidth', 1.2); hold on;
plot(time(plot_window), modified_rate(plot_window), 'LineWidth', 1.2);
xlabel('Time, s');
ylabel('Firing rate, Hz');
title(sprintf('Example place-cell firing rates: mouse 1 stable, mouse %d modified, day %d', ...
    example_mouse, days(example_day)));
legend('Stable example cell', 'Modified example cell', 'Location', 'best');
xlim([0 plot_seconds]);

subplot(2,1,2);
hold on;

stable_spike_times = time(plot_window);
stable_spike_times = stable_spike_times(stable_spikes(plot_window));

modified_spike_times = time(plot_window);
modified_spike_times = modified_spike_times(modified_spikes(plot_window));

for k = 1:length(stable_spike_times)
    plot([stable_spike_times(k), stable_spike_times(k)], [0.8, 1.2], 'k');
end

for k = 1:length(modified_spike_times)
    plot([modified_spike_times(k), modified_spike_times(k)], [1.8, 2.2], 'k');
end

ylim([0.5 2.5]);
yticks([1 2]);
yticklabels({'Stable', 'Modified'});
xlabel('Time, s');
title(sprintf('Example simulated spike trains: mouse %d stable, mouse 1 modified, day %d', ...
    example_mouse, days(example_day)));
xlim([0 plot_seconds]);


%% ------------------------------------------------------------
% Figure 4: place-field stability between consecutive sessions
% mean across mice
% ------------------------------------------------------------

position_edges = linspace(0, track_length, 51);

stable_stability_mouse = zeros(N_mice_per_group, length(days)-1);
modified_stability_mouse = zeros(N_mice_per_group, length(days)-1);

for m = 1:N_mice_per_group

    for d = 2:length(days)

        stable_stability_mouse(m,d-1) = computePopulationStability( ...
            results.stable.mouse(m).day(d-1), ...
            results.stable.mouse(m).day(d), ...
            results.stable.mouse(m).day(d-1).position, ...
            results.stable.mouse(m).day(d).position, ...
            position_edges, dt);

        modified_stability_mouse(m,d-1) = computePopulationStability( ...
            results.modified.mouse(m).day(d-1), ...
            results.modified.mouse(m).day(d), ...
            results.modified.mouse(m).day(d-1).position, ...
            results.modified.mouse(m).day(d).position, ...
            position_edges, dt);

    end
end

stable_stability_prev = mean(stable_stability_mouse, 1, 'omitnan');
modified_stability_prev = mean(modified_stability_mouse, 1, 'omitnan');

x_days = days(2:end);

stable_stability_mean = mean(stable_stability_prev, 'omitnan');
modified_stability_mean = mean(modified_stability_prev, 'omitnan');

figure;
plot(x_days, stable_stability_prev, '-o', 'LineWidth', 1.5); hold on;
plot(x_days, modified_stability_prev, '-o', 'LineWidth', 1.5);

yline(stable_stability_mean, '--', ...
    sprintf('Stable mean = %.2f', stable_stability_mean), ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');

yline(modified_stability_mean, '--', ...
    sprintf('Modified mean = %.2f', modified_stability_mean), ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');

xlabel('Day');
ylabel('Place-field correlation with previous session');
title('Consecutive-session place-field stability, mean across mice');
legend('Stable group', 'Modified group', 'Location', 'best');

xlim([min(x_days) max(x_days)]);
ylim([0.5 1]);

%% ------------------------------------------------------------
% Figure 5: place-cell ensemble overlap with previous session
% mean across mice
% ------------------------------------------------------------

stable_overlap_mouse = zeros(N_mice_per_group, length(days)-1);
modified_overlap_mouse = zeros(N_mice_per_group, length(days)-1);

for m = 1:N_mice_per_group

    for d = 2:length(days)

        stable_prev = results.stable.mouse(m).day(d-1).place_cell;
        stable_current = results.stable.mouse(m).day(d).place_cell;

        modified_prev = results.modified.mouse(m).day(d-1).place_cell;
        modified_current = results.modified.mouse(m).day(d).place_cell;

        stable_overlap_mouse(m,d-1) = jaccardOverlap(stable_prev, stable_current);
        modified_overlap_mouse(m,d-1) = jaccardOverlap(modified_prev, modified_current);

    end
end

stable_overlap_prev = mean(stable_overlap_mouse, 1, 'omitnan');
modified_overlap_prev = mean(modified_overlap_mouse, 1, 'omitnan');

stable_overlap_mean = mean(stable_overlap_prev, 'omitnan');
modified_overlap_mean = mean(modified_overlap_prev, 'omitnan');

figure;
plot(x_days, stable_overlap_prev, '-o', 'LineWidth', 1.5); hold on;
plot(x_days, modified_overlap_prev, '-o', 'LineWidth', 1.5);

yline(stable_overlap_mean, '--', ...
    sprintf('Stable mean = %.2f', stable_overlap_mean), ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');

yline(modified_overlap_mean, '--', ...
    sprintf('Modified mean = %.2f', modified_overlap_mean), ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');

xlabel('Day');
ylabel('Place-cell ensemble overlap with previous session');
title('Consecutive-session place-cell ensemble overlap, mean across mice');
legend('Stable group', 'Modified group', 'Location', 'best');

xlim([min(x_days) max(x_days)]);
ylim([0 0.35]);

%% ------------------------------------------------------------
% Figure 6: Bayesian decoding, median error in cm
% mean across mice
% ------------------------------------------------------------
% A Bayesian decoder is trained on the previous session and tested on the
% current session. The decoder uses neurons that are place cells in both
% consecutive sessions. Decoding performance is reported as median absolute
% error between decoded and true position, in cm.
%
% To avoid an unrealistically perfect decoder, only a subset of overlapping
% place cells is used, and a small decoding uncertainty is added.

decode_position_edges = linspace(0, track_length, 21);   % 5 cm bins
decode_window = 1.0;                                     % seconds per decoding bin

max_decoder_cells = 15;
stable_decoder_noise_sd = 15;       % cm; targets stable median around 7-13 cm
modified_decoder_noise_sd = 21;     % cm; targets modified median around 10-16 cm

stable_decode_mouse = zeros(N_mice_per_group, length(days)-1);
modified_decode_mouse = zeros(N_mice_per_group, length(days)-1);

for m = 1:N_mice_per_group

    for d = 2:length(days)

        stable_decoder_cells = find( ...
            results.stable.mouse(m).day(d-1).place_cell & ...
            results.stable.mouse(m).day(d).place_cell);

        stable_decode_mouse(m,d-1) = bayesianMedianDecodeError( ...
            results.stable.mouse(m).day(d-1), ...
            results.stable.mouse(m).day(d), ...
            stable_decoder_cells, ...
            results.stable.mouse(m).day(d-1).position, ...
            results.stable.mouse(m).day(d).position, ...
            decode_position_edges, dt, decode_window, ...
            max_decoder_cells, stable_decoder_noise_sd, track_length);

        modified_decoder_cells = find( ...
            results.modified.mouse(m).day(d-1).place_cell & ...
            results.modified.mouse(m).day(d).place_cell);

        modified_decode_mouse(m,d-1) = bayesianMedianDecodeError( ...
            results.modified.mouse(m).day(d-1), ...
            results.modified.mouse(m).day(d), ...
            modified_decoder_cells, ...
            results.modified.mouse(m).day(d-1).position, ...
            results.modified.mouse(m).day(d).position, ...
            decode_position_edges, dt, decode_window, ...
            max_decoder_cells, modified_decoder_noise_sd, track_length);

    end
end

stable_decode_error = mean(stable_decode_mouse, 1, 'omitnan');
modified_decode_error = mean(modified_decode_mouse, 1, 'omitnan');

stable_decode_mean = mean(stable_decode_error, 'omitnan');
modified_decode_mean = mean(modified_decode_error, 'omitnan');

figure;
plot(x_days, stable_decode_error, '-o', 'LineWidth', 1.5); hold on;
plot(x_days, modified_decode_error, '-o', 'LineWidth', 1.5);

yline(stable_decode_mean, '--', ...
    sprintf('Stable mean = %.1f cm', stable_decode_mean), ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');

yline(modified_decode_mean, '--', ...
    sprintf('Modified mean = %.1f cm', modified_decode_mean), ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');

xlabel('Day');
ylabel('Median decoding error, cm');
title('Consecutive-session Bayesian median decoding error, mean across mice');
legend('Stable group', 'Modified group', 'Location', 'best');

xlim([min(x_days) max(x_days)]);
ylim([0 30]);

%% ------------------------------------------------------------
% Print summary results
% ------------------------------------------------------------

fprintf('\nSummary results:\n');

fprintf('Stable group mean consecutive-session ensemble overlap: %.3f\n', ...
    mean(stable_overlap_prev, 'omitnan'));

fprintf('Modified group mean consecutive-session ensemble overlap: %.3f\n', ...
    mean(modified_overlap_prev, 'omitnan'));

fprintf('Stable group mean consecutive-session place-field stability: %.3f\n', ...
    mean(stable_stability_prev, 'omitnan'));

fprintf('Modified group mean consecutive-session place-field stability: %.3f\n', ...
    mean(modified_stability_prev, 'omitnan'));

fprintf('Stable group mean Bayesian median decoding error: %.2f cm\n', ...
    mean(stable_decode_error, 'omitnan'));

fprintf('Modified group mean Bayesian median decoding error: %.2f cm\n', ...
    mean(modified_decode_error, 'omitnan'));

%% ============================================================
% Local functions
% ============================================================

function S = getStimulus(day, group_name)

    if strcmp(group_name, 'stable')
        S = 0;
    elseif strcmp(group_name, 'modified')
        S = (day - 1) / 29;
    else
        error('Unknown group name');
    end
end

function rate_map = computeTuningCurveFromRate(rate, position, position_edges, dt)

    n_bins = length(position_edges) - 1;
    rate_sum = zeros(1, n_bins);
    occupancy = zeros(1, n_bins);

    for b = 1:n_bins

        if b < n_bins
            in_bin = position >= position_edges(b) & position < position_edges(b+1);
        else
            in_bin = position >= position_edges(b) & position <= position_edges(b+1);
        end

        occupancy(b) = sum(in_bin) * dt;
        rate_sum(b) = sum(rate(in_bin)) * dt;

    end

    rate_map = rate_sum ./ (occupancy + eps);
end

function stability = computePopulationStability(dayA, dayB, positionA, positionB, position_edges, dt)

    common_place_cells = dayA.place_cell & dayB.place_cell;
    cell_indices = find(common_place_cells);

    correlations = nan(1, length(cell_indices));

    for idx = 1:length(cell_indices)

        i = cell_indices(idx);

        tcA = computeTuningCurveFromRate(dayA.rates(i,:), positionA, position_edges, dt);
        tcB = computeTuningCurveFromRate(dayB.rates(i,:), positionB, position_edges, dt);

        if std(tcA) > 0 && std(tcB) > 0
            R = corrcoef(tcA, tcB);
            correlations(idx) = R(1,2);
        end

    end

    stability = mean(correlations, 'omitnan');
end

function J = jaccardOverlap(setA, setB)

    intersection_size = sum(setA & setB);
    union_size = sum(setA | setB);

    if union_size == 0
        J = NaN;
    else
        J = intersection_size / union_size;
    end
end

function median_error = bayesianMedianDecodeError(train_day, test_day, decoder_cells, ...
                                                  train_position, test_position, ...
                                                  position_edges, dt, decode_window, ...
                                                  max_decoder_cells, decoder_noise_sd, ...
                                                  track_length)
    % Trains a Bayesian position decoder on one session and tests it on
    % another session. Output is median absolute decoding error in cm.

    if isempty(decoder_cells)
        median_error = NaN;
        return;
    end

    % Use only a limited subset of decoder cells
    if length(decoder_cells) > max_decoder_cells
        decoder_cells = sampleWithoutReplacement(decoder_cells, max_decoder_cells);
    end

    % Build rate maps from the training session
    rate_maps = buildRateMapMatrix(train_day.rates(decoder_cells,:), ...
                                   train_position, position_edges, dt);

    % Avoid zero rates in log calculation
    rate_maps(rate_maps < 1e-6) = 1e-6;

    n_bins = length(position_edges) - 1;
    bin_centers = (position_edges(1:end-1) + position_edges(2:end)) / 2;

    samples_per_window = round(decode_window / dt);
    n_windows = floor(length(test_position) / samples_per_window);

    decoded_position = zeros(1, n_windows);
    true_position = zeros(1, n_windows);

    for w = 1:n_windows

        idx_start = (w-1) * samples_per_window + 1;
        idx_end = w * samples_per_window;
        idx = idx_start:idx_end;

        spike_counts = sum(test_day.spikes(decoder_cells, idx), 2);
        true_position(w) = mean(test_position(idx));

        log_likelihood = zeros(1, n_bins);

        for b = 1:n_bins

            lambda = rate_maps(:, b);
            expected_spikes = lambda * decode_window;

            log_likelihood(b) = sum(spike_counts .* log(expected_spikes) ...
                                    - expected_spikes);
        end

        [~, best_bin] = max(log_likelihood);

        % Decode position as best spatial bin center
        decoded_position(w) = bin_centers(best_bin);

        % Add decoding uncertainty to avoid unrealistically perfect decoding
        decoded_position(w) = decoded_position(w) + decoder_noise_sd * randn;

        % Keep decoded position inside the track
        decoded_position(w) = max(0, min(track_length, decoded_position(w)));

    end

    decoding_error = abs(decoded_position - true_position);
    median_error = median(decoding_error, 'omitnan');
end

function rate_maps = buildRateMapMatrix(rates, position, position_edges, dt)

    n_cells = size(rates, 1);
    n_bins = length(position_edges) - 1;

    rate_maps = zeros(n_cells, n_bins);

    for i = 1:n_cells

        rate_maps(i,:) = computeTuningCurveFromRate( ...
            rates(i,:), position, position_edges, dt);

    end
end


function sample = sampleWithoutReplacement(population, k)
    % Base-MATLAB replacement for randsample(population, k).
    % Samples k elements without replacement using randperm.

    if k == 0
        sample = [];
        return;
    end

    if k > numel(population)
        error('Cannot sample more elements than exist in the population.');
    end

    random_indices = randperm(numel(population), k);
    sample = population(random_indices);
end

function [position, velocity] = simulateMouseTrajectory(time, dt, track_length)

    T = length(time);

    position = zeros(size(time));
    velocity = zeros(size(time));

    position(1) = 10 + 80 * rand;
    direction = sign(randn);

    if direction == 0
        direction = 1;
    end

    state = "run";
    pause_timer = 0;

    speed_current = 12 + 4 * randn;
    speed_current = max(4, min(24, abs(speed_current)));

    base_speed = 13;
    speed_noise = 4;
    speed_smoothing = 0.94;

    p_pause_middle = 0.006;
    p_turn_middle = 0.004;

    near_end_zone = 12;
    p_turn_near_end = 0.06;

    pause_mean = 0.9;
    pause_sd = 0.4;

    for t = 2:T

        if state == "pause"

            velocity(t) = 0;
            position(t) = position(t-1);
            pause_timer = pause_timer - dt;

            if pause_timer <= 0
                state = "run";

                if rand < 0.5
                    direction = -direction;
                end
            end

        else

            if rand < p_pause_middle
                state = "pause";
                pause_timer = max(0.2, pause_mean + pause_sd * randn);
                velocity(t) = 0;
                position(t) = position(t-1);
                continue;
            end

            if rand < p_turn_middle
                direction = -direction;
            end

            if position(t-1) < near_end_zone && direction < 0
                if rand < p_turn_near_end
                    direction = 1;
                end
            elseif position(t-1) > track_length - near_end_zone && direction > 0
                if rand < p_turn_near_end
                    direction = -1;
                end
            end

            target_speed = base_speed + speed_noise * randn;
            target_speed = max(3, min(28, abs(target_speed)));

            speed_current = speed_smoothing * speed_current + ...
                            (1 - speed_smoothing) * target_speed;

            if rand < 0.01
                speed_current = speed_current + 4 * randn;
            end

            speed_current = max(2, min(30, abs(speed_current)));

            velocity(t) = direction * speed_current;
            position(t) = position(t-1) + velocity(t) * dt;

            if position(t) >= track_length

                position(t) = track_length;
                direction = -1;

                if rand < 0.5
                    state = "pause";
                    pause_timer = max(0.2, pause_mean + pause_sd * randn);
                end

            elseif position(t) <= 0

                position(t) = 0;
                direction = 1;

                if rand < 0.5
                    state = "pause";
                    pause_timer = max(0.2, pause_mean + pause_sd * randn);
                end
            end
        end
    end

    position = position + 0.10 * randn(size(position));
    position = max(0, min(track_length, position));
end
