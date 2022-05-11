using MLDatasets, MAT

@inline function load_MNIST_data(input_number::Int64; input_type::DataType = Float64)
    
    return vec(MLDatasets.MNIST.traintensor(input_type, input_number))
    
end

@inline function load_Faces_data(input_number::Int64; input_type::DataType = Float64)
    
    return nothing
    
end

@inline function load_Symbols_data(input_number::Int64; input_type::DataType = Float64)
    
    return nothing
    
end

@inline function load_3DObjects_data(input_number::Int64; rotation_direction::String = "x")
    "ObjectMatrix_allObjects_allRotations.mat" in readdir() ? nothing : error("object view file not in present directory")
    
    q, r = divrem(input_number, 360)
    iszero(r) ? begin object_number = q; view_number = 360 end : begin object_number = q+1; view_number = r end

    varname = string("imageMatrix_", string(object_number), "_", rotation_direction)
    file = matopen("ObjectMatrix_allObjects_allRotations.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, view_number])
    
end


function get_dataset_sepcifics(dataset::String)
    if dataset == "MNIST"
        sample = load_MNIST_data(1)
        max_inputs = 60000
        input_call_function = load_MNIST_data
        classes = 0:9

    elseif dataset == "Faces"
        sample = load_Faces_data(1)
        max_inputs = 2000
        input_call_function = load_Faces_data
        classes = nothing

    elseif dataset == "Symbols"
        sample = zeros(256)
        max_inputs = 1000
        input_call_function = load_Symbols_data
        classes = nothing

    elseif dataset == "3DObjects"
        sample = zeros(10000)
        max_inputs = 18000
        input_call_function = load_3DObjects_data
        classes = 1:50

    else
        error("dataset not recognized")

    end
    return sample, max_inputs, input_call_function, classes
end


@inline function get_input_index(dataset::String, class::Union{Int64, Array{Int64, 1}}; repeats::Int64 = 1)
    ret_index = Array{Int64, 1}(undef, repeats)

    if dataset == "MNIST"
        for r in 1:repeats
            random_start = rand(1:30000)
            class_labels = MNIST.trainlabels(random_start:random_start+500)
            for i in eachindex(class_labels) 
                class_labels[i] in class ? begin ret_index[r] = i+random_start-1; break end : nothing 
            end
        end

    elseif dataset == "3DObjects"
        random_start = rand(1:360-repeats+1)
        for r in 1:repeats ret_index[r] = 360*(class - 1) + random_start + r - 1 end
    end

    return ret_index
end


@inline function get_input(dataset::String; class::Union{Int64, Array{Int64, 1}} = zeros(Int64, 0), index::Int64 = 0, repeats::Int64 = 1)
    !isempty(class) && !iszero(index) ? error("cannot create consistent index and class") : nothing
    !iszero(index) && repeats != 1 ? begin @warn "can return only one index, reducing repeats to 1"; repeats = 1 end : nothing

    sample, max_inputs, input_call, _ = get_dataset_sepcifics(dataset)
    inputs = Array{eltype(sample), 2}(undef, length(sample), repeats)

    if !isempty(class)
        input_index = get_input_index(dataset, class, repeats = repeats)
        for i in 1:repeats inputs[:, i] = input_call(input_index[i]) end

    elseif !iszero(index)
        inputs[:, 1:repeats] .= input_call(index)

    else
        training_range = convert(Int64, ceil(0.5*max_inputs))
        inputs[:, 1:repeats] .= input_call(rand(1:training_range))

    end
    
    return inputs
end

