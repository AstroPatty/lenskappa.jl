using DataFrames
using LinearAlgebra
using Distributions
using Statistics
using Plots
using .lenskappa
struct KappaModel{T<:AbstractFit}
    fit::T
    bins::Vector{Float64}
    weights::Vector{Float64}
end

function get_covariance_matrix(variances::Vector{Float64}, correlation_matrix::Matrix{Float64})
    # Create the covariance matrix
    covariance_matrix = diagm(variances)
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
    field_weights:: DataFrame, kappa_weights::DataFrame, weights_to_use::Vector{String}, 
    variances::Vector{Float64}, tolerance::Float64 = 1.0, max_chi_squared::Float64 = 1.0,
    max_tolerance::Float64 = 20.0)
    # Inforces a maximum chi_squared
    models = Vector{KappaModel}()
    while tolerance < max_tolerance
        best_fit = model(field_weights, kappa_weights, weights_to_use, variances, tolerance)
        push!(models, best_fit)
        if best_fit.fit.χ² < max_chi_squared
            return best_fit
        elseif best_fit.fit.χ² >= max_chi_squared
            @info "Chi squared  $(best_fit.fit.χ²) > $(max_chi_squared), increasing tolerance"
            tolerance = tolerance * 1.1
            @info "Increasing tolerance to $tolerance"
        end
    end
    models = sort!(models, by = model -> model.fit.χ²)
    best_fit = models[1]
    @info "Maximum iterations reached, returning best fit with chi squared $(best_fit.fit.χ²)"
    return best_fit
end

function model(
    field_weights:: DataFrame, kappa_weights::DataFrame, weights_to_use::Vector{String}, 
    variances::Vector{Float64}, tolerance::Float64 = 1.0)
    # Fit the data to the GEV with a least squares method.
    @info "Starting model..."

    selected_field_weights = field_weights[!, weights_to_use]
    # find any rows with invalid values


    # Get the correlation matrix
    correlation_matrix = cor(Matrix(selected_field_weights))

    kappas = kappa_weights[!, :kappa]
    selected_kappa_weights = kappa_weights[!, weights_to_use]
    # Get the medians of the selected weights
    medians = median.(selected_field_weights[:,w] for w in weights_to_use)
    scaled_variances = (medians.^2) .* variances
    scaled_variances = scaled_variances .* tolerance^2
    # Create the covariance matrix
    covariance_matrix = get_covariance_matrix(scaled_variances, correlation_matrix)  
    # Get the selected lines of sight
    mask = select(selected_kappa_weights, medians, scaled_variances)
    @info "Selected $(sum(mask)) lines of sight out of $(size(selected_kappa_weights, 1))"
    best_fit = model(kappas, Matrix(selected_kappa_weights), medians, mask, covariance_matrix)
    best_fit
end

function model(
    kappas::Vector{Float64}, kappa_weights::Matrix{Float64}, medians::Vector{Float64},
    mask::BitVector, covariance_matrix::Matrix{Float64}, bin_width::Float64 = 0.01)
    
    # create the multivaritate normal distribution
    mvn = MultivariateNormal(medians, covariance_matrix)
    # create bins from smallest to largest weight values
    weights_to_use = kappa_weights[mask, :]
    kappas_to_use = kappas[mask]

    kappa_bins = collect(-0.2:0.002:0.5)
    kappa_storage = zeros(length(kappa_bins) - 1)

    mins = minimum.(eachcol(weights_to_use))
    maxs = maximum.(eachcol(weights_to_use))


    bins = Tuple(collect(min:bin_width:max) for (min, max) in zip(mins, maxs))


    bin_centers = ((bin[1:end-1] + bin[2:end]) ./ 2 for bin in bins)
    # Create array of indices for each weight
    bin_indices = Iterators.product(Tuple(1:length(b) for b in bins)...)
 
    bins = [searchsortedfirst.(Ref(bin), weight) for (bin, weight) in zip(bins, eachcol(weights_to_use))]
    kappa_storage = zeros(length(kappa_bins) - 1, length(bin_indices))

    # create the bin centers
    Threads.@threads for (i_, idx) in collect(enumerate(bin_indices))
        bin_mask = trues(length(kappas_to_use))
        for (i, bn) in enumerate(idx)
            bin_mask = bin_mask .& (bins[i] .== bn)
        end 
        kappas_in_bin = kappas_to_use[bin_mask]
        if length(kappas_in_bin) == 0
            continue
        end
        bin_weight_center = [mean(weight[bin_mask]) for weight in eachcol(weights_to_use)]
        pdf_ = Distributions.pdf(mvn, bin_weight_center)
        # Bin the kappas 
        kappa_histogram = Distributions.fit(Histogram, kappas_in_bin, kappa_bins).weights
        kappa_histogram = kappa_histogram ./ length(kappas_in_bin)
        kappa_histogram = kappa_histogram .* pdf_
        # put this in the appropriate column 
        kappa_storage[:, i_] = kappa_histogram
    end
    # sum the columns
    kappa_storage = sum(kappa_storage, dims=2)
    kappa_storage = dropdims(kappa_storage, dims=2)
    densities = kappa_storage.*(kappa_bins[2:end] - kappa_bins[1:end-1])
    densities = kappa_storage ./ sum(densities)
    # safe the figure

    best_fit = fit(kappa_bins, densities)
    best_fit = KappaModel(best_fit, kappa_bins, densities)

    return best_fit
end


function select(
    weight_data::DataFrame, medians::Vector{Float64}, variances::Vector{Float64}, width::Float64 = 5.0)

    standard_deviations = sqrt.(variances)
    # Get the medians of the selected weights
    # Find the weights that are within the width of the median
    mask = trues(size(weight_data, 1))
    for (i, median) in enumerate(medians)
        mask = mask .& (abs.(weight_data[:,i] .- median) .< (width .* standard_deviations[i]))
    end
    return mask
end