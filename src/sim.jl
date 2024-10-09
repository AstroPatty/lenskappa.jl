include("kappa.jl")
include("weights.jl")
using DataFrames
using CSV
using StatsBase

struct SimulatedMap
    weights::SimulationWeights
    kappa::kappas
end

function SimulatedMap(weights::Matrix{Float32}, kappa::kappas, names::Vector{String})::SimulatedMap
    # Make sure they are the same length
    if length(weights) != length(kappa)
        @error "Weights and kappa must be the same length"
    end
    # Find the location of any rows with NaNs or infs
    bad_rows = (isnan.(weights) .| isinf.(weights))
    if (any(bad_rows))
        @warn "Removing" sum(bad_rows) " rows with NaNs or Infs"
        weights = weights[.!bad_rows, :]
    end
    sim_weights = SimulationWeights(weights, names)


    return SimulatedMap(sim_weights, kappa)
end

function SimulatedMap(path::String, weights::Vector{String})
    # Check that the path is a directory and exists
    if !isdir(path)
        @error path * " does not exist or is not a directory"
    end

    # find CSV files in the directory
    files = filter(x -> endswith(x, ".csv"), readdir(path))
    if length(files) == 0
        @error "Directory " * path * " does not contain any CSV files"
    end
    # Load the CSVs and concatenate into a single DataFrame
    @info "Loading CSV files in folder " * path
    dataframes = [CSV.read(joinpath(path, f), DataFrame) for f in files]
    @info "Concatenating CSV files"
    values = DataFrame()
    try
        values = vcat(dataframes...)
    catch
        @error "Could not concatenate the CSV files. Are their columns the same?"
    end
    # Extract the weights
    SimulatedMap(values, weights)


    # Extract the kappa values
end

function SimulatedMap(values::DataFrame, weights::Vector{String})
    # Extract the weights
    w = SimulationWeights(values, weights)
    # Extract the kappa values
    kappas = try
        values[!, :kappa]
    catch
        @error "Could not find a kappa column in the DataFrame"
    end
    k = kappa(kappas)
    return SimulatedMap(w, k)
end


function get_kappas_by_range(sim::SimulatedMap, ranges::Vararg{WeightRange,N})::Vector{Float32} where {N}
    weight_indices = []
    try
        weight_indices = [findfirst(sim.weights.names .== r.weight) for r in ranges]
    catch
        @error "The weights expected in the ranges are not in this map..."
    end
    mask = trues(length(sim.kappa))
    for i in weight_indices
        mask = mask .& (sim.weights.weight_values[:, i] .>= ranges[i].min) .& (sim.weights.weight_values[:, i] .<= ranges[i].max)
    end
    return @view sim.kappa[mask]
end


function make_kappa_histogram(sim::SimulatedMap, bins::Vector{Float64}, ranges::Vararg{WeightRange,N})::Histogram where {N}
    kappas = get_kappas_by_range(sim, ranges...)
    fit(Histogram, kappas, bins)
end
