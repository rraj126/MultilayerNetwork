function update_layer_dims!(nlayers::Int64, layer_dims::Array{Int64, 1})

    if isempty(layer_dims)
        for i in 1:nlayers push!(layer_dims, 1000*i) end
    else
        nlayers == length(layer_dims) && all(layer_dims .> 0) ? nothing : error("incorrect layer specifications")
    end 

    return nothing
end

function create_network(usr_args::Dict{Symbol, Any})
    nlayers = get!(usr_args, :nlayers, 1)
    isa(nlayers, Int64) && nlayers > 0 ? nothing : error("number of layers must be a positive integer")

    layer_dims = get!(usr_args, :layer_dims, Array{Int64, 1}(undef, 0))
    update_layer_dims!(nlayers, layer_dims)

    return nothing
end

function modulate(c::Vector{Float64}, d::Vector{Float64}; modulation_factor::Float64 = 0.5)
    length(c) == length(d) ? nothing : error("vectors must have same length for modulation")

    c_contribution = modulation_factor .* (c .> 0.0)
    iszero(d) ? d_contribution = zeros(length(d)) : d_contribution = (1 - modulation_factor) .* (d ./ maximum(d))

    return c_contribution + d_contribution
end