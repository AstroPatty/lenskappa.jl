using DataFrames

abstract type AbstractWeights end

struct SimulationWeights <: AbstractWeights
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

struct ObservationalWeights <: AbstractWeights
    weight_values::Array{Float64,2}
    names::Array{String,1}
end

struct WeightRange
    weight::String
    range::Tuple{Float64,Float64}
    WeightRange(range::Tuple{Float64,Float64}, weight::String) = range[1] < range[2] ? new(range, weight) : new((range[2], range[1]), weight)
end

function WeightRange(min::Float64, max::Float64, weight::String)::WeightRange
    return WeightRange((min, max), weight)
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

    weight_values = convert(Array{Float64,2}, df[:, names])
    # Remove rows with NaNs and Infs
    weight_values = weight_values[.!any(isnan.(weight_values), dims=2), :]
    return ObservationalWeights(weight_values, names)
end

