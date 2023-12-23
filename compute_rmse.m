function rmse = compute_rmse(U, V, p, q, data)
    n = size(data, 1);
    error_sum = 0;
    for i = 1:n
        user_id = data(i, 1);
        movie_id = data(i, 2);
        rating = data(i, 3);
        error = U(user_id, :) * V(movie_id, :)' + p(user_id) + q(movie_id) - rating;
        error_sum = error_sum + error^2;
    end
    rmse = sqrt(error_sum / n);
end