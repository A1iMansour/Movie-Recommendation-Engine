%%% Data Pre-processing %%%

% Step 1: Import data from "MovieLens_Dataset.xlsx"
ratings_data = readmatrix('MovieLens_Dataset.xlsx', 'Sheet', 'Ratings');
users_data = readcell('MovieLens_Dataset.xlsx', 'Sheet', 'Users', 'Range', 'A2:D944');
movies_data = readcell('MovieLens_Dataset.xlsx', 'Sheet', 'Movies', 'Range', 'A2:U1683');
genres_data = readcell('MovieLens_Dataset.xlsx', 'Sheet', 'Genres', 'Range', 'A2:B20');

% Step 2: Shuffle and split data into training and testing sets
shuffled_indices = randperm(size(ratings_data, 1));
ratings_data = ratings_data(shuffled_indices, :);

training_ratio = 0.8;
training_size = floor(training_ratio * size(ratings_data, 1));

training_data = ratings_data(1:training_size, :);
testing_data = ratings_data(training_size+1:end, :);

num_iterations = 10000000;
alpha = 0.001;

%%% Optimization %%%

% Step 3: Initialize variables
r = 10; % Maximum feasible rank
U = rand(943, r); % User matrix
V = rand(1682, r); % Movie matrix
p = rand(943, 1); % User bias vector
q = rand(1682, 1); % Movie bias vector

% Step 4: Stochastic gradient descent optimization
lambda = 0.1; % Regularization parameter

for j = 1:num_iterations
    % Randomly select a data point
    random_index = randi(size(training_data, 1));
    user_id = training_data(random_index, 1);
    movie_id = training_data(random_index, 2);
    rating = training_data(random_index, 3);

    % Calculate error terms
    error = 2 * (U(user_id, :) * V(movie_id, :)' + p(user_id) + q(movie_id) - rating);
    
    % Update U, V, p, q
    U(user_id, :) = U(user_id, :) - alpha * (error * V(movie_id, :) + 2 * lambda * U(user_id, :));
    V(movie_id, :) = V(movie_id, :) - alpha * (error * U(user_id, :) + 2 * lambda * V(movie_id, :));
    p(user_id) = p(user_id) - alpha * (error);
    q(movie_id) = q(movie_id) - alpha * (error);
end


%%% Evaluation %%%

% Initialization
r = [1, 3, 5, 7, 9];
alpha = 0.001;
lambda_values = [0.001, 0.01, 0.1, 1];

figure;
plotIndex = 1; % Index to keep track of subplot number
% SGD Optimization

for i = 1:length(r)
    for k = 1:length(lambda_values)
        % Initialize U, V, p, and q
        U = rand(size(users_data, 1), r(i));
        V = rand(size(movies_data, 1), r(i));
        p = rand(size(users_data, 1), 1);
        q = rand(size(movies_data, 1), 1);

        rmse_training = zeros(num_iterations/10000, 1);
        rmse_testing = zeros(num_iterations/10000, 1);

        for j = 1:num_iterations
            idx = randi(size(training_data, 1));

            user_id = training_data(idx, 1);
            movie_id = training_data(idx, 2);
            rating = training_data(idx, 3);

            % Calculate error terms
            error = 2 * (U(user_id, :) * V(movie_id, :)' + p(user_id) + q(movie_id) - rating);
            
            % Update U, V, p, q
            U(user_id, :) = U(user_id, :) - alpha * (error * V(movie_id, :) + 2 * lambda_values(k) * U(user_id, :));
            V(movie_id, :) = V(movie_id, :) - alpha * (error * U(user_id, :) + 2 * lambda_values(k) * V(movie_id, :));
            p(user_id) = p(user_id) - alpha * (error);
            q(movie_id) = q(movie_id) - alpha * (error);

            % Evaluate RMSE
            if mod(j, 10000) == 0
                rmse_training(j/10000) = compute_rmse(U, V, p, q, training_data);
                rmse_testing(j/10000) = compute_rmse(U, V, p, q, testing_data);
            end
        end

        subplot(length(r), length(lambda_values), plotIndex); % Create a subplot
        plot(10000:10000:num_iterations, rmse_training, 'r-', 'LineWidth', 2);
        hold on;
        plot(10000:10000:num_iterations, rmse_testing, 'b-', 'LineWidth', 2);
        title(['r = ', num2str(r(i)), ', lambda = ', num2str(lambda_values(k))]);
        xlabel('Iterations');
        ylabel('RMSE');
        legend('Training Data', 'Testing Data');
        grid on;
        hold off;

        plotIndex = plotIndex + 1; % Increment plot index
    end
end

%%% Feature Exraction %%%
r = 9;
lambda = 0.1;
U = rand(size(users_data, 1), r);
V = rand(size(movies_data, 1), r);
p = rand(size(users_data, 1), 1);
q = rand(size(movies_data, 1), 1);

for j = 1:num_iterations
    idx = randi(size(training_data, 1));

    user_id = training_data(idx, 1);
    movie_id = training_data(idx, 2);
    rating = training_data(idx, 3);

    % Calculate error terms
    error = 2 * (U(user_id, :) * V(movie_id, :)' + p(user_id) + q(movie_id) - rating);
    
    % Update U, V, p, q
    U(user_id, :) = U(user_id, :) - alpha * (error * V(movie_id, :) + 2 * lambda * U(user_id, :));
    V(movie_id, :) = V(movie_id, :) - alpha * (error * U(user_id, :) + 2 * lambda * V(movie_id, :));
    p(user_id) = p(user_id) - alpha * (error);
    q(movie_id) = q(movie_id) - alpha * (error);
end

% Perform feature selection for 3 different features
feature_indices = [2,6,8];

for i = 1:numel(feature_indices)
    feature_index = feature_indices(i);
    
    % Sort column of V corresponding to the feature in descending order
    [~, sorted_movies] = sort(V(:, feature_index), 'descend');

    % Select movies with at least 25 ratings
    fprintf('Top Movie for Feature %d:\n', feature_index);
    nb_movies_per_feature = 0;
    all_genres_of_feature = [];
    for k = 1:size(sorted_movies)
        if nb_movies_per_feature >= 5
            break;
        end

        nb_movie_ratings = 0;
        
        for l = 1:size(ratings_data)
            if(ratings_data(l,2) == sorted_movies(k))
                nb_movie_ratings=nb_movie_ratings+1;
            end
        end
        if (nb_movie_ratings >= 25)
            nb_movies_per_feature=nb_movies_per_feature+1;
            % Assuming selected_movies is a cell array
            selected_movie_genres = movies_data(sorted_movies(k), 3:end);
            selected_movie_id = sorted_movies(k);
            selected_movie_name= movies_data(sorted_movies(k),2);

            % Get the genres of the selected movies      
            selected_genres = find(cell2mat(selected_movie_genres) == 1);
            selected_genres_names = getNamesFromIDs(selected_genres, genres_data);
            all_genres_of_feature = [all_genres_of_feature, selected_genres_names];
            fprintf('Movie ID: %s, Movie Name: %s, Genres: %s\n\n', string(selected_movie_id), string(selected_movie_name), strjoin(selected_genres_names, ', '));
        end
    end
    fprintf('All feature genres: %s\n\n', strjoin(unique(all_genres_of_feature), ', '));
end

%%% Recommending Similar Movies %%%
ratingsCount = 0;
while ratingsCount<25
    ratingsCount = 0;
    movieIndex = randi([1, 1682]); % generate a random integer between 1 and 1682
    for i = 1:size(ratings_data, 1)
        if(ratings_data(i,2) == movieIndex)
            ratingsCount=ratingsCount+1;
        end
    end
end


for j = 1:1682
    % Compute cosine similarity
    similarity(j) = dot(V(movieIndex,:), V(j,:)) / (norm(V(movieIndex,:)) * norm(V(j,:)));
end

topSimilarMovies = top5Indices(similarity);
for i=1:6
    topSimilarMoviesNames(i) = movies_data(topSimilarMovies(i),2);
end

fprintf('Top Similar Movies to %s are: %s\n\n', topSimilarMoviesNames{1}, strjoin(topSimilarMoviesNames(2:end), ', '));


%%% Fair Recommendation Engine %%%

% Define the rank and regularization parameter
r = 9;
lambda = 0.1;

% Initialize user and movie matrices with random values
U = rand(size(users_data, 1), r);
V = rand(size(movies_data, 1), r);

% Initialize user and movie bias vectors with random values
p = rand(size(users_data, 1), 1);
q = rand(size(movies_data, 1), 1);

% Perform stochastic gradient descent
for j = 1:num_iterations
    % Randomly select a training example
    idx = randi(size(training_data, 1));

    % Extract user ID, movie ID, and rating from the selected training example
    user_id = training_data(idx, 1);
    movie_id = training_data(idx, 2);
    rating = training_data(idx, 3);

    % Calculate error term
    error = 2 * (U(user_id, :) * V(movie_id, :)' + p(user_id) + q(movie_id) - rating);
    
    % Update user and movie matrices and bias vectors
    U(user_id, :) = U(user_id, :) - alpha * (error * V(movie_id, :) + 2 * lambda * U(user_id, :));
    V(movie_id, :) = V(movie_id, :) - alpha * (error * U(user_id, :) + 2 * lambda * V(movie_id, :));
    p(user_id) = p(user_id) - alpha * (error);
    q(movie_id) = q(movie_id) - alpha * (error);
end

% Extract the gender column from the user data
gender = users_data(:, 3);

% Separate users by gender
usersm = users_data(strcmp(gender, 'M'), :);  % Male users
usersf = users_data(strcmp(gender, 'F'), :);  % Female users

% Convert user IDs to numeric format
userIDs_m = cell2mat(usersm(:, 1));
userIDs_f= cell2mat(usersf(:,1));

% Find corresponding rows in user matrix and bias vector for each gender
rows_in_U_m = U(userIDs_m, :);
rows_in_U_f = U(userIDs_f, :);
rows_in_pm = p(userIDs_m, :);
rows_in_pf = p(userIDs_f, :);

% Extract genre columns from the movie data
column_act = cell2mat(movies_data(:, 4));  % Action
column_mu = cell2mat(movies_data(:, 15));  % Musical
column_rom = cell2mat(movies_data(:, 17));  % Romance
column_sci = cell2mat(movies_data(:, 18));  % Science Fiction

% Find rows where the corresponding genre column has a value of 1
rowsact = find(column_act == 1);
rowsmu = find(column_mu == 1);
rowsrom = find(column_rom == 1);
rowssci = find(column_sci == 1);

% Extract corresponding movie IDs
idsact= cell2mat(movies_data(rowsact, 1));
idsmu= cell2mat(movies_data(rowsmu, 1));
idsrom= cell2mat(movies_data(rowsrom, 1));
idssci= cell2mat(movies_data(rowssci, 1));

% Find corresponding rows in movie matrix and bias vector for each genre
rows_in_sc = V(idssci, :);
rows_in_act = V(idsact, :);
rows_in_rom = V(idsrom, :);
rows_in_mus = V(idsmu, :);

rows_in_scq = q(idssci, :);
rows_in_actq = q(idsact, :);
rows_in_romq = q(idsrom, :);
rows_in_musq = q(idsmu, :);

% Calculate predicted ratings for each gender and genre
% The formula used is: X = u*v + p + q
% The bsxfun function is used to add the bias vector to each row of the rating matrix

% Male users
ratings_rom_m = bsxfun(@plus, rows_in_U_m * rows_in_rom' + rows_in_pm, rows_in_romq');
ratings_mu_m = bsxfun(@plus, rows_in_U_m * rows_in_mus' + rows_in_pm, rows_in_musq');
ratings_sci_m = bsxfun(@plus, rows_in_U_m * rows_in_sc' + rows_in_pm, rows_in_scq');
ratings_act_m = bsxfun(@plus, rows_in_U_m * rows_in_act' + rows_in_pm, rows_in_actq');

% Female users
ratings_rom_f = bsxfun(@plus, rows_in_U_f * rows_in_rom' + rows_in_pf, rows_in_romq');
ratings_mu_f = bsxfun(@plus, rows_in_U_f * rows_in_mus' + rows_in_pf, rows_in_musq');
ratings_sci_f = bsxfun(@plus, rows_in_U_f * rows_in_sc' + rows_in_pf, rows_in_scq');
ratings_act_f = bsxfun(@plus, rows_in_U_f * rows_in_act' + rows_in_pf, rows_in_actq');

% Calculate average ratings for each gender and genre
values = {'Gender', 'Romance','Action', 'Sci', 'musical'; 
    'Male', mean(ratings_rom_m(:)), mean(ratings_act_m(:)), mean(ratings_sci_m(:)), mean(ratings_mu_m(:));
    'Female', mean(ratings_rom_f(:)), mean(ratings_act_f(:)), mean(ratings_sci_f(:)), mean(ratings_mu_f(:))};

% Display the results
disp(values)

%IMPOSE FAIRNESS

Um = 4*numel(usersm); % Total number of male users multiplied by 4 (4 genres)
Uf = 4*numel(usersf); % Total number of female users multiplied by 4 (4 genres)
Gg= numel(rowsact) + numel(rowsmu) + numel(rowsrom)+ numel(rowssci); % Total number of genres

r = 9; % Maximum feasible rank
U = rand(943, r); % Initialize User matrix with random values
V = rand(1682, r); % Initialize Movie matrix with random values
p = rand(943, 1); % Initialize User bias vector with random values
q = rand(1682, 1); % Initialize Movie bias vector with random values
alpha = 0.01; % Set learning rate
num_iterations = 10000000; % Set number of iterations
lambda2_values = [0.001, 0.01, 0.1, 1]; % Set regularization parameters
lambda=0.1; % Set regularization parameter
figure; % Create a new figure
plotIndex = 1; % Initialize plot index

%Start Stochastic Gradient Descent (SGD) Optimization
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
movies_inG= cat(1,idssci,idsmu,idsrom,idsact);

for k = 1:length(lambda2_values)
        % Initialize U, V, p, and q with random values
        U = rand(size(users_data, 1), r);
        V = rand(size(movies_data, 1), r);
        p = rand(size(users_data, 1), 1);
        q = rand(size(movies_data, 1), 1);
        rmse_training = zeros(num_iterations/10000, 1); % Initialize RMSE for training data
        rmse_testing = zeros(num_iterations/10000, 1); % Initialize RMSE for testing data

        % Start iterations
        for j = 1:num_iterations
            idx = randi(size(training_data, 1)); % Pick a random index from training data
            user_id = training_data(idx, 1); % Get user id
            movie_id = training_data(idx, 2); % Get movie id
            rating = training_data(idx, 3); % Get rating

            % Calculate error terms
            error = 2 * (U(user_id, :) * V(movie_id, :)' + p(user_id) + q(movie_id) - rating);

            % Calculating ui * vj + pi, since it is used multiple times 
            uvp=U(user_id, :) * V(movie_id, :)' + p(user_id);

            isMale = ismember(user_id, userIDs_m);

            if (isMale)
                % Update U, V, p, q using SGD for males
                
                % Update U using Stochastic Gradient Descent
                U(user_id, :) = U(user_id, :) - alpha * (error * V(movie_id, :) + 2 * lambda * U(user_id, :) + 2 * lambda2_values(k) * (uvp / (Um * Gg))) .* (V(movie_id, :) / (Um * Gg));
            
                % Update V using Stochastic Gradient Descent
                V(movie_id, :) = V(movie_id, :) - alpha * (error * U(user_id, :) + 2 * lambda * V(movie_id, :) + 2 * lambda2_values(k) * (1 / (Um * Gg) * uvp) .* (U(user_id, :) / (Um * Gg)));
            
                % Update p using Stochastic Gradient Descent
                p(user_id) = p(user_id) - alpha * (error + lambda2_values(k) * 2 * (1 / (Um * Gg) * uvp - 1 / (Uf * Gg) * uvp)) .* (1 * (size(Um, 1) + size(Gg, 1))) / (Um * Gg);
            
                % Update q using Stochastic Gradient Descent
                q(movie_id) = q(movie_id) - alpha * (error);
            
            else
                % Update U, V, p, q using SGD for females
            
                % Update U using Stochastic Gradient Descent
                U(user_id, :) = U(user_id, :) - alpha * (error * V(movie_id, :) + 2 * lambda * U(user_id, :) + 2 * lambda2_values(k) * (-uvp / (Uf * Gg))) .* ((-V(movie_id, :) / (Uf * Gg)));
            
                % Update V using Stochastic Gradient Descent
                V(movie_id, :) = V(movie_id, :) - alpha * (error * U(user_id, :) + 2 * lambda * V(movie_id, :) + 2 * lambda2_values(k) * (-1 / (Uf * Gg) * uvp)) .* (-U(user_id, :) / (Uf * Gg));
            
                % Update p using Stochastic Gradient Descent
                p(user_id) = p(user_id) - alpha * (error + lambda2_values(k) * 2 * (1 / (Um * Gg) * uvp - 1 / (Uf * Gg) * uvp)) .* (-(1 * (size(Uf, 1) + size(Gg, 1)) / (Uf * Gg)));
            
                % Update q using Stochastic Gradient Descent
                q(movie_id) = q(movie_id) - alpha * (error);
            end


            % Update U, V, p, q using SGD
            U(user_id, :) = U(user_id, :) - alpha * (  error * V(movie_id, :) + 2 * lambda * U(user_id, :) + 2 * lambda2_values(k) * (uvp/(Um*Gg) - uvp/(Uf*Gg))) .* ((V(movie_id, :)/(Um*Gg) -V(movie_id, :)/(Uf*Gg)));

            V(movie_id, :) = V(movie_id, :) - alpha * (error * U(user_id, :) + 2 * lambda * V(movie_id, :)+ 2 * lambda2_values(k) * (1/(Um*Gg)*uvp -1/(Uf*Gg)*uvp)) .* ((U(user_id, :)/(Um*Gg) -U(user_id, :)/(Uf*Gg)));
            
            p(user_id) = p(user_id) - alpha * (error+ lambda2_values(k)*2*(1/(Um*Gg)*uvp -1/(Uf*Gg)*uvp)) .* ((1 * (size(Um,1) + size(Gg,1)))/(Um*Gg) - (1 * (size(Uf,1) + size(Gg,1))/(Uf*Gg)));

            q(movie_id) = q(movie_id) - alpha * (error);

            % Evaluate RMSE every 10000 iterations
            if mod(j, 10000) == 0
                rmse_training(j/10000) = compute_rmse(U, V, p, q, training_data);
                rmse_testing(j/10000) = compute_rmse(U, V, p, q, testing_data);
            end
        end

        % Plot RMSE for training and testing data
        subplot(length(r), length(lambda2_values), plotIndex); % Create a subplot
        plot(10000:10000:num_iterations, rmse_training, 'r-', 'LineWidth', 2); % Plot RMSE for training data
        hold on;
        plot(10000:10000:num_iterations, rmse_testing, 'b-', 'LineWidth', 2); % Plot RMSE for testing data
        title(['r = ', num2str(r), ', lambda_2 = ', num2str(lambda2_values(k))]); % Set title
        xlabel('Iterations'); % Set x-label
        ylabel('RMSE'); % Set y-label
        legend('Training Data', 'Testing Data'); % Set legend
        grid on;
        hold off;

        plotIndex = plotIndex + 1; % Increment plot index

        % Find corresponding rows in matrix U and p for male and female users
        rows_in_U_m = U(userIDs_m, :);
        rows_in_U_f = U(userIDs_f, :);
        rows_in_pm = p(userIDs_m, :);
        rows_in_pf = p(userIDs_f, :);

        % Find corresponding rows in matrix V and q for each genre
        rows_in_sc = V(idssci, :);
        rows_in_act = V(idsact, :);
        rows_in_rom = V(idsrom, :);
        rows_in_mus = V(idsmu, :);
        
        rows_in_scq = q(idssci, :);
        rows_in_actq = q(idsact, :);
        rows_in_romq = q(idsrom, :);
        rows_in_musq = q(idsmu, :);

        % Calculate ratings for each genre and gender using the formula: X = u*v + p + q
        % The following code calculates ratings for each genre and gender and stores them in respective variables

        % Display lambda2_values(k) and the calculated ratings
        disp(lambda2_values(k))
        values = {'Gender', 'Romance','Action', 'Sci', 'musical'; 
            'Male', mean(ratings_rom_m(:)), mean(ratings_act_m(:)), mean(ratings_sci_m(:)), mean(ratings_mu_m(:));
            'Female', mean(ratings_rom_f(:)), mean(ratings_act_f(:)), mean(ratings_sci_f(:)), mean(ratings_mu_f(:))};
        disp(values)
end