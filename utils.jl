"""
    extract(A, measurement_series, measurement_series_idx)

Extracts rows from matrix `A` that belong to the measurement series with the
index given in `measurement_series_idx` and using the mapping from measurement 
point to series provided in `measurement_series`.

##### Arguments
- `A`: Matrix from which rows will be extracted.
- `measurement_series`: The mapping from measurement point to series
- `measurement_series_idx`: Index used to filter rows based on the condition.

##### Returns
A new matrix containing only the rows that belong to the measurement series.
"""
function extract(A, measurement_series, measurement_series_idx)
    mask = (measurement_series .== measurement_series_idx)
    return extract_via_mask(A, mask)
end


"""
    extract_via_mask(A, mask)

Extracts rows from matrix `A` based on the provided boolean `mask`.

##### Arguments
- `A`: Matrix from which rows will be extracted.
- `mask`: Boolean mask used to filter rows.

##### Returns
A new matrix containing only the rows that satisfy the condition.
"""
function extract_via_mask(A, mask)
    @assert size(A, 1) == size(mask, 1) "The first dimension between `A` and `mask` must match"
    n = sum(mask)
    m = size(A, 2)
    return reshape(A[repeat(mask, 1, m)], (n, m))
end


"""
    plot_at_idx(p, A, measurement_series, label; measurement_series_idx, feature_idx)

Plots the measurement series specified by `measurement_series_idx` of the feature
specified by `feature_idx` for the given data matrix `A`.

##### Arguments
- `p`: Plot object to which the data will be added.
- `A`: Data matrix.
- `measurement_series`: The mapping from measurement point to series
- `label`: Matrix of labels.
- `measurement_series_idx`: Index of the measurement series.
- `feature_idx`: Selects the feature to plot.

##### Returns
A plot object with the selected data added.
"""
function plot_at_idx(p, A, measurement_series, label; measurement_series_idx, feature_idx)
    
    masked_data = extract(A, measurement_series, measurement_series_idx)[:, feature_idx]
    return plot!(p, masked_data, label=label[:,feature_idx])
end


"""
    plot_histogram(A, names; feature_idx)

Plots a histogram for a specific feature in the whole dataset.

##### Arguments
- `A`: Data matrix.
- `names`: Array of feature names.
- `feature_idx`: Index of the feature to plot.

##### Returns
A plot object with the histogram.
"""
function plot_histogram(A, names; feature_idx)

    name = names[feature_idx]

    histogram(A[:, feature_idx], bins=50, xlabel="bins", ylabel="Count", label="$name", title="Feature $feature_idx", alpha=0.7)

    return plot!(legend=true)
end
