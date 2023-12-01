
using Optim
using Distributions
using StatsBase

abstract type AbstractFit end

struct GumbelFit <: AbstractFit
    μ::Float64
    β::Float64
    χ²::Float64
end

struct GeVFit <: AbstractFit
    μ::Float64
    β::Float64
    ξ::Float64
    χ²::Float64
end

function GumbelFit(bins::Array{Float64, 1}, values::Array{Float64,1}, tolerance::Float64 = 0.05)
    # Fit the data to the Gumbel with a least squares method.
    # compute the bin bin_centers
    bin_centers = (bins[2:end] + bins[1:end-1]) ./ 2
    # Get the bin widths
    # Define the objective function
    function objective(parameters::Array{Float64,1})
        μ, logβ = parameters
        gumbel = Distributions.Gumbel(μ, exp(logβ))
        gumbel = pdf.(gumbel, bin_centers)
        # normalize by sum and bin width
        squared_deviations = (values - gumbel).^2
        return sum(squared_deviations)        
    end

    # Define the initial guess
    initial_guess = [0.0, -3]
    # Define the optimization problem
    problem = optimize(objective, initial_guess, LBFGS())
    # Get the solution
    solution = Optim.minimizer(problem)
    chi_squared = Optim.minimum(problem)
    reduced_chi_squared = chi_squared / (length(values) - 2)
    return GumbelFit(solution..., reduced_chi_squared)
end

function GeVFit(bins::Array{Float64,1}, values::Array{Float64,1}, tolerance::Float64 = 0.05)
    # Fit the data to the GEV with a least squares method.
    # compute the bin bin_centers
    bin_centers = (bins[2:end] + bins[1:end-1]) ./ 2
    # Get the bin widths
    # Define the objective function
    uncertainties = abs.(tolerance .* values)
    function objective(parameters::Array{Float64,1})
        μ, logβ, ξ = parameters
        gev = Distributions.GeneralizedExtremeValue(μ, exp(logβ), ξ)
        gev = pdf.(gev, bin_centers)
        # normalize by sum and bin width
        squared_deviations = ((values - gev)/uncertainties).^2
        return sum(squared_deviations)        
    end

    # Define the initial guess
    initial_guess = [0.0, -3, 0.0]
    # Define the optimization problem
    problem = optimize(objective, initial_guess, LBFGS())
    # Get the solution
    solution = Optim.minimizer(problem)
    chi_squared = Optim.minimum(problem)
    reduced_chi_squared = chi_squared / (length(values) - 3)
    return GeVFit(solution..., reduced_chi_squared)


end

function fit(bins::Array{Float64,1}, values::Array{<:Real,1}, type::Symbol = :GEV, tolerance::Float64 = 0.05)
    if length(bins) != length(values) + 1
        throw(ArgumentError("The number of bins must be one more than the number of values"))
    end
    densitys = values.*(bins[2:end] - bins[1:end-1])
    total = sum(densitys)
    # normalize by the bin sum and the bin width

    values = values ./ total

    if type == :Gumbel
        return GumbelFit(bins, values, tolerance)
    elseif type == :GEV
        return GeVFit(bins, values, tolerance)
    else
        throw(ArgumentError("Unknown fit type $type"))
    end
end

fit(data::Array{Float64,2},; type::Symbol = :GEV) = fit(type, data[:,1], data[:,2], type = type)

function fit(kappas::Array{Float64, 1}, n_bins::Int64, type::Symbol = :GEV, tolerance::Float64 = 0.05)
    # Compute the histogram
    bins = collect(range(minimum(kappas), stop=maximum(kappas), length=n_bins+1))
    values = StatsBase.fit(StatsBase.Histogram, kappas, bins).weights
    return fit(bins, values, type, tolerance)
end