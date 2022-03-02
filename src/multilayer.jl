function update_layer_dims!(nlayers::Int64, layer_dims::Array{Int64, 1})

    if isempty(layer_dims)
        for i in 1:nlayers push!(layer_dims, 1000*i) end
    else
        nlayers == length(layer_dims) && all(layer_dims .> 0) ? nothing : error("incorrect layer specifications")
    end 

    return nothing
end

function update_lambda!(lambda::Array{Float64, 1}, layer_dims::Array{Int64, 1})
    if isempty(lambda)
        min_size = minimum(layer_dims)
        for layer in eachindex(layer_dims) push!(lambda, 0.1*layer_dims[layer]/min_size) end

    else
        length(lambda) == length(layer_dims) ? nothing : error("lambda must be specified for each layer")
    end

    return nothing
end

function update_modulation_factor!(modulation_factor::Array{Float64, 1}, nlayers::Int64)
    if isempty(modulation_factor)
        if nlayers > 1
            for _ in 1:nlayers-1 push!(modulation_factor, 0.9) end
        end
    else
        length(modulation_factor) == nlayers-1 ? nothing : error("modulation factors must be specified between all layers")
    end

    return nothing
end

function update_signal_cap!(signal_cap::Array{Float64, 1}, nlayers::Int64)
    if isempty(signal_cap)
        if nlayers > 1
            for _ in 1:nlayers-1 push!(signal_cap, 0.4) end
        end
    else
        length(signal_cap) == nlayers-1 ? nothing : error("signal cap must be specified for all layers beyond first")
    end

    return nothing
end

function update_connection_sparseness!(connection_sparseness::Array{Float64, 1}, nlayers::Int64)
    if isempty(connection_sparseness)
        if nlayers > 1
            for _ in 1:nlayers push!(connection_sparseness, 0.01) end
        end
    else
        length(connection_sparseness) == nlayers ? nothing : error("connection_sparseness must be specified for all layers")
    end

    return nothing
end


function create_network(usr_args::Dict{Symbol, Any})
    nlayers = get!(usr_args, :nlayers, 1)
    isa(nlayers, Int64) && nlayers > 0 ? nothing : error("number of layers must be a positive integer")

    layer_dims = get!(usr_args, :layer_dims, Array{Int64, 1}(undef, 0))
    update_layer_dims!(nlayers, layer_dims)

    lambda = get!(usr_args, :lambda, Array{Float64, 1}(undef, 0))
    update_lambda!(lambda, layer_dims)

    modulation_factor = get!(usr_args, :modulation_factor, Array{Float64, 1}(undef, 0))
    update_modulation_factor!(modulation_factor, nlayers)

    signal_cap = get!(usr_args, :signal_cap, Array{Float64, 1}(undef, 0))
    update_signal_cap!(signal_cap, nlayers)

    connection_sparseness = get!(usr_args, :connection_sparseness, Array{Float64, 1}(undef, 0))
    update_connection_sparseness!(connection_sparseness, nlayers)

    return nothing
end

function modulate(c::Vector{Float64}, d::Vector{Float64}; modulation_factor::Float64 = 0.5, signal_cap::Float64 = 0.4)
    length(c) == length(d) ? nothing : error("vectors must have same length for modulation")
    terms_to_null = convert(Int64, round((1.0 - signal_cap)*length(c)))
    
    c_trunc = copy(c)
    ind = sortperm(c)
    for i in 1:terms_to_null c_trunc[ind[i]] = 0.0 end

    d_trunc = copy(d)
    ind = sortperm(d)
    for i in 1:terms_to_null d_trunc[ind[i]] = 0.0 end

    c_contribution = modulation_factor .* (c_trunc .> 0.0)
    d_contribution = (1 - modulation_factor) .* (d_trunc .> 0.0)


    #c_contribution = modulation_factor .* (c .> 0.0)
    #iszero(d) ? d_contribution = zeros(length(d)) : d_contribution = (1 - modulation_factor) .* (d ./ maximum(d))
    #d_contribution = (1 - modulation_factor) .* (d .> 0.0)

    return c_contribution + d_contribution
end