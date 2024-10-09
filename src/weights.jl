using DataFrames

abstract type AbstractCounts end

struct SimulationWeights <: AbstractCounts
    weight_values::Array{Float64,2}
    names::Array{String,1}
    function SimulationWeights(weight_values::Array{Float64,2}, names::Array{String,1})
        if length(names) != length(unique(names))
            error("Duplicate names found in weights")
        end

        # NaNs and Infs will be handled at a higher level, so fail if they are present
        if any(isnan.(weight_values)) || any(isinf.(weight_values))
            error("NaNs or Infs found in weights")
        end

        new(weight_values, names)
    end
end

struct ObservationalWeights <: AbstractCounts
    weight_values::Array{Float64,2}
    names::Array{String,1}
end

struct WeightRange
    min::Number
    max::Number
    weight::String
    WeightRange(min::Number, max::Number, weight::String) = min < max ? new(min, max, weight) : new(max, min, weight)
end


function WeightRange(range::Tuple{Float64,2}, weight::String)::WeightRange
    return WeightRange(range[1], range[2], weight)
end

function SimulationWeights(df::DataFrame, names::Array{String,1})::SimulationWeights
    # Check for duplicate names
    weight_values = Matrix{Float64}(df[:, names])
    return SimulationWeights(weight_values, names)
end

function ObservationalWeights(df::DataFrame, column_names::Array{String,1})::ObservationalWeights
    # Check for duplicate names
    if length(column_names) != length(unique(column_names))
        error("Duplicate names found in weights")
    end
    # print column names
    weight_values = Matrix{Float64}(df[:, column_names])
    # Make a mask of rows with NaNs or Infs
    mask = vec(any(isnan.(weight_values), dims=2) .| any(isinf.(weight_values), dims=2))
    weight_values = weight_values[.!mask, :]

    return ObservationalWeights(weight_values, column_names)
end

function ObservationalWeights(path::String, names::Array{String,1})::ObservationalWeights
    @info "Reading observational weights from $path"
    if !isfile(path) || !endswith(path, ".csv")
        @error "File not found or not a CSV"
    end
    weight_data = CSV.read(path, DataFrame)
    ObservationalWeights(weight_data, names)
end


function get_weights_in_box(Weights::ObservationalWeights, ranges::Vararg{WeightRange,N})::ObservationalWeights where {N}
    mask = trues(size(Weights.weight_values, 1))
    for weight_range in ranges
        column_index = findfirst(Weights.names .== weight_range.weight)
        mask = mask .& (weight_range.min .<= Weights.weight_values[:, column_index] .<= weight_range.max)
    end
    if all(mask)
        return Weights
    end
    values = @view Weights.weight_values[mask, :]
    return ObservationalWeights(values, Weights.names)
end

function make_density(weights::ObservationalWeights, bins_per_dim::Int, ranges::Vararg{WeightRange,N}) where {N}
    trucated_weights = get_weights_in_box(weights, ranges...)
    bins_edges = map(w -> range(w.min, w.max, length=bins_per_dim), ranges)
    # create boxes
    weight_ranges = map(
        (edges, r) -> map(
            (min, max) -> WeightRange(min, max, r.weight),
            edges[1:end-1], edges[2:end]
        ), bins_edges, ranges
    )
    weight_boxes = Iterators.product(weight_ranges...)
    # count the number of weights in each box
    counts = map(
        (box) -> get_weights_in_box(trucated_weights, box...),
        weight_boxes
    )
    # normalize the counts
    total = sum(map(c -> size(c.weight_values, 1), counts))
    densities = map(c -> size(c.weight_values, 1) / total, counts)
    return weight_boxes, densities

end

