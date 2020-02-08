%% Homework 1

% Name: Pantelis Dimitroulis
% UTSA ID: tkk918
% Course: EE6363--Deep Learning, Spring 2019

clc; clear; close all;

% Read data from file
file_data = importdata('EEG_driving_data_sample.mat');

brain_data = []; % channels x brain bands => 30 x 2
n_channels = 30;

% Class 0 - 30 channels
for i=1:n_channels
    brain_data = [brain_data; extract_channel_features(file_data.data_class_0(i,:,:), 250), 0];
    brain_data = [brain_data; extract_channel_features(file_data.data_class_1(i,:,:), 250), 1];
end

%% plot data
class0 = find(brain_data(:,3)==0);
class1 = find(brain_data(:,3)==1);
figure(1);
subplot(1,2,1);
plot(brain_data(class0, 1), brain_data(class0, 2), '*', 'Color', 'r'); hold on;
plot(brain_data(class1, 1), brain_data(class1, 2), '*', 'Color', 'b');
xlabel('Theta'); ylabel('Alpha')
title('Data points');
grid on; grid minor;

%% Standardizing data
feature_means_array = sum(brain_data(:,1:end-1))/length(brain_data(:,1));
std_array = std(brain_data(:,1:end-1));
brain_data(:,1:end-1) = (brain_data(:,1:end-1) - feature_means_array)./std_array;

%% plot data
figure(1);
subplot(1,2,2);
plot(brain_data(class0, 1), brain_data(class0, 2), '*', 'Color', 'r'); hold on;
plot(brain_data(class1, 1), brain_data(class1, 2), '*', 'Color', 'b');
xlabel('Theta'); ylabel('Alpha')
title('Standardized data points');
grid on; grid minor;


%% --- Algo (Logistic Regression) ---

% shuffle data rows
shuffled_data = brain_data(randperm(size(brain_data,1)),:);

%% Create folders
n_folders = 5;
len_folder = length(brain_data)/n_folders;

data = shuffled_data;

% Parameters (Log Regr)
lr = 0.3; % learning rate
threshold = 0.5;
w = [0 0 0]'; w0 = 0; % weights

% Print parameters table
parameters_table = table(lr,threshold, w0, w(1), w(2), w(3));
parameters_table.Properties.VariableNames = {'Learning_Rate' 'Threshold' 'w0' 'w1' 'w2' 'w3'};
disp(parameters_table)

% Set Variables
y = data(:,end); % labels
x = data'; % columns = samples
[sample_size feature_size] = size(data);

% Functions
z = @(i,w,w0) w' * x(:,i) + w0; % i for each sample
s = @(i,w,w0) 1/(1+exp(-z(i,w,w0)));
x2 = @(x1,w1,w2) (-w1*x1-w0)/w2; % graphical boundary

% Initialize values
cost_array = [];
sensitivity = 0;
specificity = 0;
accuracy = 0;

%% --- 5-fold cross-validation ---

for f = 1:n_folders
    testing_index_start = (f-1)*len_folder+1;
    testing_index_end = f*len_folder;
    training_indexes = [1:testing_index_start-1, testing_index_end+1:sample_size];

    % -- Training --
    % Cost function
    C = 0;
    for i = training_indexes
        C = C + (-1/sample_size) * (y(i)*log(s(i,w,w0)) + (1-y(i))*log(1-s(i,w,w0))); % cost function
    end
    cost_array = [cost_array C];
    % Gradient of cost function
    grad_C = [];
    for j = 1:feature_size
        dC_dw = 0;
        for i = training_indexes
            dC_dw = dC_dw + x(j,i)*(s(i,w,w0)-y(i));
        end
        grad_C = [grad_C; dC_dw];
    end
    % Derivative w/ respect to w0
    dC_dw0 = 0;
    for i = training_indexes
        dC_dw0 = dC_dw0 + s(i,w,w0)-y(i);
    end
    % Update weights
    w = w - lr.*grad_C;
    w0 = w0 - lr*dC_dw0;
    disp(['Cost = ' num2str(C)])

    % -- Evaluation --
    true_pos = 0;
    true_neg = 0;
    false_pos = 0;
    false_neg = 0;
    for i = testing_index_start:testing_index_end
        if y(i)==1
            if s(i,w,w0)>=threshold
                true_pos = true_pos + 1;
            else
                false_neg = false_neg + 1;
            end
        elseif y(i)==0
            if s(i,w,w0)>=threshold
                false_pos = false_pos + 1;
            else
                true_neg = true_neg + 1;
            end
        end
    end
    sensitivity = sensitivity + true_pos/(true_pos+false_neg);
    specificity = specificity + true_neg/(true_neg+false_pos);
    accuracy = accuracy + (true_pos+true_neg)/len_folder;
end

%% Mean of measurements
sensitivity = sensitivity/n_folders;
specificity = specificity/n_folders;
accuracy = accuracy/n_folders;

performance_table = table(sensitivity,specificity,accuracy);
disp(performance_table)

% Plot boundary line
figure(1);
subplot(1,2,2);
plot(-3:0.2:3, x2(-3:0.2:3,w(1),w(2)), 'Color', 'black');
grid on; grid minor;

% Plot cost function
figure(2);
subplot(1,2,1);
plot(1:n_folders, cost_array);
xlabel('trainings'); ylabel('cost');
title('Cost function')
grid on; grid minor;

%% -- Compute ROC curve ---
true_pos = 0;
true_neg = 0;
false_pos = 0;
false_neg = 0;
sensitivity_array = [];
specificity_array = [];

for threshold = [0:0.1:1]
    for i = testing_index_start:testing_index_end
        if y(i)==1
            if s(i,w,w0)>=threshold
                true_pos = true_pos + 1;
            else
                false_neg = false_neg + 1;
            end
        elseif y(i)==0
            if s(i,w,w0)>=threshold
                false_pos = false_pos + 1;
            else
                true_neg = true_neg + 1;
            end
        end
    end
    sensitivity_array = [sensitivity_array true_pos/(true_pos+false_neg)];
    specificity_array = [specificity_array true_neg/(true_neg+false_pos)];
end

% compute AUC (area under the curve)
auc = abs(trapz(1-specificity_array, sensitivity_array));
disp(['AUC = ', num2str(auc)]);

% Plot ROC curve
figure(2)
subplot(1,2,2);
plot(1-specificity_array, sensitivity_array);
xlabel('FPR (1-specificity)'); ylabel('TPR (sensitivity)');
title('ROC curve')
text(mean(1-specificity_array), mean(sensitivity_array), strcat('AUC = ', num2str(auc)) );
grid on; grid minor;

%--- END of FILE ---
