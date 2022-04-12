using MultivariateStats
using LinearAlgebra

function generate_randomized_connections(m::Int64, n::Int64, connection_sparseness::Float64)
    m < n ? error("initial connections require dimension expansion") : nothing

    seed = 1.0 .* (rand(m, n) .< connection_sparseness)
    F = svd(seed)
    size(F.U, 2) == n ? randomized_connections = F.U : randomized_connections = F.U[:, 1:n]

    #seed = rand(m, m)
    #SymMatrix = seed'*seed
    #eigenVectors = eigvecs(SymMatrix)
    #randomized_connections = col_normalize(eigenVectors[:, 1:n], showNorms = false)

    return randomized_connections 
end


function generate_decorrelation_matrix(DATA::AbstractArray{<:Real, 2}, variance_cutoff::Float64)
    PCobject = fit(PCA, DATA)
    SCORE = projection(PCobject)
    LATENT = principalvars(PCobject)

    Q, colNorms = col_normalize(SCORE, showNorms = true)
    lambda = inv.((colNorms.*colNorms).*LATENT)

    number_of_PCs = 0
    variance_explained = 0
    for component_no = 1:length(LATENT)
        variance_explained = sum(LATENT[1:component_no])./sum(LATENT)
        variance_explained >= variance_cutoff ? begin number_of_PCs = component_no; break end : nothing
    end

    print("Explained variance : $variance_explained, Number of PCs : $number_of_PCs\n")
    return Diagonal(lambda[1:number_of_PCs]), Q[:, 1:number_of_PCs]
end


function initialize_feedforward_connections(dataset::String, layer_dims::Array{Int64, 1}, connection_sparseness::Array{Float64, 1})
    fraction = 0.25
    variance_cutoff = 0.96
    nlayers = length(layer_dims)

    sample, max_inputs, input_call = get_dataset_sepcifics(dataset)
    sample_size = convert(Int64, round(fraction*max_inputs))

    DATA = Array{eltype(sample), 2}(undef, length(sample), sample_size)
    for i in 1:sample_size
        print_progress("fetching initialization inputs...", i, sample_size)  
        DATA[:, i] = input_call(rand(1:max_inputs)) 
    end

    D, Q = generate_decorrelation_matrix(DATA, variance_cutoff)
    D .= D.^(2.0^(-nlayers))

    W = Array{Array{Float64, 2}, 1}(undef, nlayers)
    left_matrix = Q
    for layer in 1:nlayers
        right_matrix = generate_randomized_connections(layer_dims[layer], size(D, 1), connection_sparseness[layer])
        W[layer] = col_normalize(left_matrix*D*transpose(right_matrix), showNorms = false)

        left_matrix = right_matrix
    end

    return W
end


function initialize_lateral_connections(max_inhib::Array{Float64, 1}, layer_dims::Array{Int64, 1})
    nlayers = length(layer_dims)
    w_lateral = Array{Array{Float64, 2}, 1}(undef, nlayers)
    for layer in 1:nlayers w_lateral[layer] = max_inhib[layer] .* ones(layer_dims[layer], layer_dims[layer]) end

    return w_lateral
end


function initialize_connection_matrix(layer_dims::Array{Int64, 1})
    nlayers = length(layer_dims)
    connection_matrix = Array{Array{Float64, 2}, 1}(undef, nlayers)
    for layer in 1:nlayers connection_matrix[layer] = zeros(layer_dims[layer], layer_dims[layer]) end

    return connection_matrix
end


function initialize_input_matrix(dataset::String, number_of_inputs::Int64)
    sample, _, _ = get_dataset_sepcifics(dataset)
    return Array{Float64, 2}(undef, length(sample), number_of_inputs)
end


function initialize_representation(number_of_inputs::Int64, network_projections::Array{Array{Float64, 2}, 1})
    nlayers = length(network_projections)
    representation_c = Array{Array{Float64, 2}, 1}(undef, nlayers)
    representation_d = Array{Array{Float64, 2}, 1}(undef, nlayers)

    for layer in 1:nlayers
        layer_dim = size(network_projections[layer], 2)
        representation_c[layer] = zeros(layer_dim, number_of_inputs)
        representation_d[layer] = zeros(layer_dim, number_of_inputs)

    end

    return representation_c, representation_d
end