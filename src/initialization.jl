using MultivariateStats
using LinearAlgebra

function generate_randomized_connections(m::Int64, n::Int64)
    m > n ? error("random connection requires dimension expansion") : nothing

    seed = rand(n, n)
    SymMatrix = seed'*seed
    eigenVectors = eigvecs(SymMatrix)
    randomized_connections = col_normalize(eigenVectors[:, 1:m], showNorms = false)

    return convert(Array{Float64, 2}, transpose(randomized_connections)) 
end


function generate_decorrelation_matrix(DATA::AbstractArray{<:Real, 2}, REP_DIM::Int64, variance_cutoff::Float64, condition_number_cutoff::Float64)

    PCobject = fit(PCA, DATA)
    SCORE = projection(PCobject)
    LATENT = principalvars(PCobject)

    Q, colNorms = col_normalize(SCORE, showNorms = true)
    lambda = inv.(sqrt.((colNorms.*colNorms).*LATENT))

    number_of_PCs = 0
    variance_explained = 0
    print("Estimating variance... \n")
    for component_no = 1:length(LATENT)
        variance_explained = sum(LATENT[1:component_no])./sum(LATENT)
        condition_number = minimum(lambda[1:component_no])/maximum(lambda[1:component_no])
        if variance_explained >= variance_cutoff || condition_number < condition_number_cutoff
            number_of_PCs = component_no
            break
        end
    end

    number_of_PCs > REP_DIM ? error(string("layer size should be greater than ", string(number_of_PCs))) : nothing

    print("Randomizing projection... \n")
    M = Diagonal(lambda[1:number_of_PCs])*Q[:, 1:number_of_PCs]'

    randomized_connections = generate_randomized_connections(number_of_PCs, REP_DIM)
    pre_projection = M'*randomized_connections
    proj = col_normalize(pre_projection, showNorms = false)

    print(string("Explained variance : ", string(round(variance_explained, digits = 2)), ", Number of PCs : ", string(number_of_PCs, "\n")))

    return proj

end


function initialize_feedforward_connections(dataset::String, dim1::Int64, dim2::Int64)
    fraction = 0.25
    variance_cutoff = 0.96
    condition_number_cutoff = 0.0001

    sample, max_inputs, input_call = get_dataset_sepcifics(dataset)
    sample_size = convert(Int64, round(fraction*max_inputs))

    DATA = Array{eltype(sample), 2}(undef, length(sample), sample_size)

    if iszero(dim1)
        for i in 1:sample_size
            print_progress("fetching random inputs for initialization...", i, sample_size)  
            DATA[:, i] = input_call(rand(1:max_inputs)) 
        end
        proj = generate_decorrelation_matrix(DATA, dim2, variance_cutoff, condition_number_cutoff)

    else
        proj = col_normalize(generate_randomized_connections(dim1, dim2))
    end

    return proj
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

initialize_lateral_connections(dim::Int64, max_inhib::Float64) = max_inhib .* ones(dim, dim)
initialize_connection_matrix(dim::Int64) = zeros(dim, dim)