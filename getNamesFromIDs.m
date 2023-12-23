function names = getNamesFromIDs(idArray, dataMatrix)
    % idArray: Input array of IDs
    % dataMatrix: Matrix with two columns (name, id)

    % Initialize the result cell array
    names = cell(size(idArray));

    % Loop through each ID in the input array
    for i = 1:numel(idArray)
        % Find the index of the ID in the dataMatrix
        idx = find(cellfun(@(x) isequal(x, idArray(i) - 1), dataMatrix(:, 2)));

        % Check if the ID exists in the dataMatrix
        if ~isempty(idx)
            % Assign the corresponding name to the result array
            names{i} = dataMatrix{idx, 1};
        else
            % If the ID is not found, you can choose to handle this case
            % Maybe assign a placeholder or an indication that the ID was not found
            names{i} = 'ID Not Found';
        end
    end
end
