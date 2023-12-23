function topIndices = top5Indices(inputArray)
    % Sort the array in descending order
    [~, sortedIndices] = sort(inputArray, 'descend');

    % Extract the top 5 indices
    topIndices = sortedIndices(1:6); % The output will consist of six movies, with the initial movie matching the one for which we are seeking similar recommendations.
end
