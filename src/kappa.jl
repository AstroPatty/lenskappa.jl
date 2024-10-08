
const MIN_KAPPA = -0.2
const MAX_KAPPA = 1.0

# Kappa values outside this range would be quite strange!

const kappas = Vector{Float64}

function kappa(vals::Vector{Float64})::kappas
    if any(vals .< MIN_KAPPA) || any(vals .> MAX_KAPPA)
        @warn "Kappa values outside the expected range $MIN_KAPPA to $MAX_KAPPA"
    end
    return vals
end
