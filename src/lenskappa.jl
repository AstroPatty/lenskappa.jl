module lenskappa

include("inference.jl")

export
    kappas,
    SimulatedMap,
    ObservationalWeights,
    WeightRange,
    get_kappas_by_range,
    make_kappa_histogram,
    make_density,
    do_inference,
    InferenceData
end
