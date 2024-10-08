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
    min::Float64
    max::Float64
    weight::String
    WeightRange(min::Float64, max::Float64, weight::String) = min < max ? new(min, max, weight) : new(max, min, weight)
end

function WeightRange(range::Tuple{Float64,2}, weight::String)::WeightRange
    return WeightRange(range[1], range[2], weight)
end

function SimulationWeights(df::DataFrame, names::Array{String,1})::SimulationWeights
    # Check for duplicate names
    weight_values = Matrix{Float64}(df[:, names])
    return SimulationWeights(weight_values, names)
end

function ObservationalWeights(df::DataFrame, names::Array{String,1})::ObservationalWeights
    # Check for duplicate names
    if length(names) != length(unique(names))
        error("Duplicate names found in weights")
    end

    weight_values = Matrix{Float64}(df[:, names])
    # Remove rows with NaNs and Infs
    weight_values = weight_values[.!any(isnan.(weight_values), dims=2), :]
    return ObservationalWeights(weight_values, names)
end

function ObservationalWeights(path::String, names::Array{String,1})::ObservationalWeights
    @info "Reading observational weights from $path"
    if !isfile(path) || !endswith(path, ".csv")
        @error "File not found or not a CSV"
    end
    weight_data = CSV.read(path, DataFrame)
    ObservationalWeights(weight_data, names)
end


