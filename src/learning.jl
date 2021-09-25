using LinearAlgebra, JLD, FileIO


function get_organization_parameters(usr_args::Dict{Symbol, Any})
    organize_interval = get!(usr_args, :organize_interval, 500)
    break_points = usr_args[:break_points]

    break_points[1] >= organize_interval ? nothing : error("inappropriate break_points")
    for bp in break_points iszero(rem(bp, organize_interval)) ? nothing : error("break_points must be multiples of organize_interval") end

    return div(break_points[end], organize_interval), organize_interval

end

function get_init_connections(usr_args::Dict{Symbol, Any})
    max_inhib = get!(usr_args, :max_inhib, 10.0)
    nlayers = usr_args[:nlayers]
    layer_dims = usr_args[:layer_dims]

    print("initializing connections... \n")

    if haskey(usr_args, :pfid)
        W = load_file(usr_args[:pfid], "projection", "p")
    else
        W = initialize_feedforward_connections(usr_args[:dataset], layer_dims)
        save_state(W, "projection", "p")
    end

    w_lateral = initialize_lateral_connections(max_inhib, layer_dims)
    connection_matrix = initialize_connection_matrix(layer_dims)

    init_proj = Array{Array{Float64, 2}, 1}(undef, nlayers)
    for layer in 1:nlayers init_proj[layer] = copy(W[layer]) end

    return W, w_lateral, connection_matrix, init_proj
end


@inline function SynPot_to_matrix!(SynPot::Dict{Tuple{Int64, Int64}, Int64}, connection_matrix::Array{Float64, 2}, w_lateral::Array{Float64, 2}, epoch::Int64)
    for k in keys(SynPot)
        connection_matrix[CartesianIndex(k)] += sign(SynPot[k])
        connection_matrix[CartesianIndex(reverse(k))] += sign(SynPot[k])

    end
    connection_matrix .= connection_matrix .* (connection_matrix .> 0.0)

    max_inhib = maximum(w_lateral)

    copyto!(w_lateral, connection_matrix)
    w_lateral .= w_lateral ./ epoch
    w_lateral .= max_inhib .* (w_lateral .<= 0) .- w_lateral

    return nothing

end

function update_lateral_connections!(SynPot::Array{Dict{Tuple{Int64, Int64}, Int64}, 1}, epoch::Int64, connections...)
    w_lateral = connections[1]
    connection_matrix = connections[2]
    
    length(SynPot) == length(w_lateral) == length(connection_matrix) ? nothing : error("lateral connections cannot be updated")
    
    nlayers = length(w_lateral)
    for layer in 1:nlayers SynPot_to_matrix!(SynPot[layer], connection_matrix[layer], w_lateral[layer], epoch) end

    return nothing

end
    

function multilayer_learning_loop(input_array::UnitRange{Int64}, usr_args::Dict{Symbol, Any}, connections...)
    mode = get!(usr_args, :mode, 1)
    randomized = get!(usr_args, :randomized, true)
    verbose = get!(usr_args, :verbose, false)

    nlayers = usr_args[:nlayers]
    W = connections[1]
    w_lateral = connections[2]
    init_proj = connections[3]

    SynPot = Array{Dict{Tuple{Int64, Int64}, Int64}, 1}(undef, nlayers)
    for i in 1:nlayers SynPot[i] = Dict{Tuple{Int64, Int64}, Int64}() end


    for inputNo in input_array
        randomized ? input = get_input(usr_args[:dataset]) : input = get_input(usr_args[:dataset], inputNo)

        for layer in 1:nlayers
            W_updated, d_response = simulate_layer(W[layer], input, lambda = usr_args[:lambda][layer], mode = mode, verbose = verbose)
            c_response = self_organize_layer(W[layer], init_proj[layer], input, SynPot = SynPot[layer], w_lateral = w_lateral[layer], lambda = usr_args[:lambda][layer])

            W[layer] = W_updated
            nlayers > 1 && layer < nlayers ? input = modulate(c_response, d_response, modulation_factor = usr_args[:modulation_factor][layer]) : nothing
        end
    end

    return SynPot

end


function run_network(usr_args::Dict{Symbol, Any})
    break_points = usr_args[:break_points]
    bp_index = 1

    nepochs, organize_interval = get_organization_parameters(usr_args)
    W, w_lateral, connection_matrix, init_proj = get_init_connections(usr_args)
    save_state(W, string(0), "d")
    save_state(w_lateral, string(0), "c")

    for i in 1:nepochs
        input_array = (i-1)*organize_interval+1:i*organize_interval
        
        SynPot = multilayer_learning_loop(input_array, usr_args, W, w_lateral, init_proj)
        update_lateral_connections!(SynPot, i, w_lateral, connection_matrix)

        if break_points[bp_index] == input_array[end]
            save_state(W, string(bp_index), "d")
            save_state(w_lateral, string(bp_index), "c")
            bp_index += 1 
        
        end 
        print_progress("learning data and classes...", i, nepochs)
    end
    save_state_sequences()
    save_parameters(usr_args)

    return nothing

end


function learn_data(dataset::String, break_points::Union{Int64, StepRange, Vector{Int64}}; kwargs...)
    usr_args = Dict{Symbol, Any}(kwargs)

    usr_args[:dataset] = dataset
    issorted(break_points) ? usr_args[:break_points] = break_points : error("break points unsorted")

    create_network(usr_args)
    run_network(usr_args)

    return nothing
end