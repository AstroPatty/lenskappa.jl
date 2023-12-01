using DataFrames
using LinearAlgebra
using Distributions
using Statistics
include("fit.jl")

function covariance_matrix(variances::Vector{Float64}, correlation_matrix::Matrix{Float64})
    # Create the covariance matrix
    covariance_matrix = Diagonal(variances)
    for i in 1:size(correlation_matrix, 1)
        for j in 1:size(correlation_matrix, 2)
            if i != j
                covariance_matrix[i,j] = correlation_matrix[i,j] * sqrt(variances[i] * variances[j])
            end
        end
    end
    return covariance_matrix
end

function model(
    weight_data::DataFrame, weights_to_use::Vector{String}, variances::Vector{Float64},
    tolerance::Float64 = 0.05)
    # Fit the data to the GEV with a least squares method.

    weights = weight_data[!, weights_to_use]
    # Get the correlation matrix
    correlation_matrix = cor(weights)


    kappas = weight_data[!, :kappa]
    # Get the medians of the selected weights
    medians = median.(weights, dims=1)
    # Create the covariance matrix
    covariance_matrix = covariance_matrix(variances, correlation_matrix)  
    # Get the selected lines of sight
    mask = select(weight_data, variances)
    model = model(kappas, weights, mask, covariance_matrix)
end

function model(kappas::Vector{Float64}, weights::DataFrame, mask::Vector{Bool}, covariance_matrix::Matrix{Float64}, bin_width::Float64 = 0.05)
    medians = median.(weights, dims=1)
    # create the multivaritate normal distribution
    mvn = MultivariateNormal(medians, covariance_matrix)
    # create bins from smallest to largest weight values
    weights_to_use = weights[mask, :]
    kappas_to_use = kappas[mask]

    kappa_bins = minimum(kappas_to_use):0.005:maximum(kappas_to_use)
    kappa_storage = zeros(length(kappa_bins) - 1)

    mins = minimum.(eachcol(weights_to_use))
    maxs = maximum.(eachcol(weights_to_use))
    bins = (min:bin_width:max for (min, max) in zip(mins, maxs))
    bin_centers = (bin[1:end-1] + bin[2:end]) ./ 2
    # Create array of indices for each weight
    bins = [searchsortedfirst.(Ref(bin), weight) for (bin, weight) in zip(bins, eachcol(weights_to_use))]

    bin_indices = Iterators.product((1: length(bin) for bin in bins)...) 
    # create the bin centers
    for idx in bin_indices
        mask = trues(length(kappas_to_use))
        for (i, bn) in enumerate(idx)
            mask = mask & (bins[i] .== bn)
        end 
        kapps_in_bin = kappas_to_use[mask]
        if length(kappas_in_bin) == 0
            continue
        end
        bin_weight_center = bin_centers[idx]
        pdf = pdf(mvn, bin_weight_center)
        # Bin the kappas 
        kappa_histogram = Distributions.fit(Histogram, kapps_in_bin, kappa_bins)
        kappa_histogram = normalize(kappa_histogram, mode=:density)
        kappa_storage .+= kappa_histogram.weights .* pdf
    end
    
    return kappa_bins, kappa_storage
end

function select(
    weight_data::DataFrame, variances::Vector{Float64}, width::Float64 = 5.0)

    standard_deviations = sqrt.(variances)
    # Get the medians of the selected weights
    medians = median.(weights, dims=1)
    # Find the weights that are within the width of the median
    mask = abs.(eachcol(weight_data) .- medians) .< (width .* standard_deviations)
    return mask
end