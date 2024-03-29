using MLDatasets, MAT

@inline function load_MNIST_data(input_number::Int64; input_type::DataType = Float64)
    
    return vec(MLDatasets.MNIST.traintensor(input_type, input_number))
    
end

@inline function load_Faces_data(input_number::Int64; noise_level::Int64 = 1)
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "FaceMatrix_All_saltnpepper.mat" in readdir(data_dir, join = false) ? nothing : error("noisy face file not in data directory")

    varname = string("subject_", string(input_number))
    file = matopen("$data_dir/FaceMatrix_All_saltnpepper.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[noise_level][:, rand(1:100)])
    
end

@inline function load_Symbols_data(input_number::Int64)
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "SymbolsMatrix_large.mat" in readdir(data_dir, join = false) ? nothing : error("large symbols file not in data directory")

    varname = "SymbolsMatrix_large"
    file = matopen("$data_dir/SymbolsMatrix_large.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, input_number])
    
end

@inline function load_SymbolsFacesCombined_data(input_number::Int64)
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "SymbolsFaces_combined.mat" in readdir(data_dir, join = false) ? nothing : error("combined symbols and face file not in data directory")

    varname = "SymbolsFaces_combined"
    file = matopen("$data_dir/SymbolsFaces_combined.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, input_number])
    
end

@inline function load_3DObjects_oldrotation_data(input_number::Int64; rotation_direction::String = "x")
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "ObjectMatrix_allObjects_allRotations.mat" in readdir(data_dir, join = false) ? nothing : error("object view file not in data directory")
    
    q, r = divrem(input_number, 360)
    iszero(r) ? begin object_number = q; view_number = 360 end : begin object_number = q+1; view_number = r end

    varname = string("imageMatrix_", string(object_number), "_", rotation_direction)
    file = matopen("$data_dir/ObjectMatrix_allObjects_allRotations.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, view_number])

end
   
@inline function load_3DObjects_rotation_data(input_number::Int64)
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "ObjectMatrix_allObjects_allRotations.mat" in readdir(data_dir, join = false) ? nothing : error("object view file not in data directory")
    
    q, r = divrem(input_number, 360)
    iszero(r) ? begin object_number = q; view_number = 360 end : begin object_number = q+1; view_number = r end

    varname = string("imageMatrix_", string(object_number), "_r")
    file = matopen("$data_dir/ObjectMatrix_allObjects_allRotations.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, view_number])

end

@inline function load_3DObjects_size_data(input_number::Int64)
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "ObjectMatrix_allObjects_allSizes.mat" in readdir(data_dir, join = false) ? nothing : error("object view file not in data directory")
    
    q, r = divrem(input_number, 360)
    iszero(r) ? begin object_number = q; view_number = 360 end : begin object_number = q+1; view_number = r end

    varname = string("imageMatrix_", string(object_number), "_s")
    file = matopen("$data_dir/ObjectMatrix_allObjects_allSizes.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, view_number])

end

@inline function load_3DObjects_translation_data(input_number::Int64; rotation_direction::String = "x")
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "ObjectMatrix_allObjects_allPositions.mat" in readdir(data_dir, join = false) ? nothing : error("object view file not in data directory")
    
    q, r = divrem(input_number, 360)
    iszero(r) ? begin object_number = q; view_number = 360 end : begin object_number = q+1; view_number = r end

    varname = string("imageMatrix_", string(object_number), "_t")
    file = matopen("$data_dir/ObjectMatrix_allObjects_allPositions.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, view_number])

end

@inline function load_Sequence_data(input_number::Int64)
    data_dir = string(pwd(), "/Data")
    isdir(data_dir) ? nothing : error("no directory named Data found")
    "Sequences.mat" in readdir(data_dir, join = false) ? nothing : error("sequence file not in data directory")

    varname = "Sequences"
    file = matopen("$data_dir/Sequences.mat")
    matrix = read(file, varname)
    close(file)
    
    return vec(matrix[:, input_number])
    
end

@inline function load_custom_data(input_number::Int64, path_to_data::String, varname::String)
    isfile(path_to_data) ? nothing : error("incorrect data path")

    file = matopen(path_to_data)
    input = read(file, varname)
    close(file)

    return input[:, input_number]
end


function get_dataset_sepcifics(dataset::String)
    if dataset == "MNIST"
        sample = load_MNIST_data(1)
        max_inputs = 60000
        input_call_function = load_MNIST_data
        classes = 0:9

    elseif dataset == "Faces"
        sample = zeros(10000)
        max_inputs = 2000
        input_call_function = load_Faces_data
        classes = 0:1

    elseif dataset == "Symbols"
        sample = zeros(10000)
        max_inputs = 1000
        input_call_function = load_Symbols_data
        classes = 1:1

    elseif dataset == "SymbolsFacesCombined"
        sample = zeros(10000)
        max_inputs = 3000
        input_call_function = load_SymbolsFacesCombined_data
        classes = 1:1

    elseif dataset == "3DObjects_r"
        sample = zeros(10000)
        max_inputs = 18000
        input_call_function = load_3DObjects_rotation_data
        classes = 1:50

    elseif dataset == "3DObjects_s"
        sample = zeros(10000)
        max_inputs = 18000
        input_call_function = load_3DObjects_size_data
        classes = 1:50

    elseif dataset == "3DObjects_t"
        sample = zeros(10000)
        max_inputs = 18000
        input_call_function = load_3DObjects_translation_data
        classes = 1:50

    elseif dataset == "Sequences"
        sample = zeros(48)
        max_inputs = 18596
        input_call_function = load_Sequence_data
        classes = 1:1

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

    elseif dataset == "3DObjects_r"
        random_start = rand(1:360-repeats+1)
        for r in 1:repeats ret_index[r] = 360*(class - 1) + random_start + r - 1 end

    elseif dataset == "3DObjects_s"
        random_start = rand(1:360-repeats+1)
        for r in 1:repeats ret_index[r] = 360*(class - 1) + random_start + r - 1 end

    elseif dataset == "3DObjects_t"
        random_start = rand(1:360-repeats+1)
        for r in 1:repeats ret_index[r] = 360*(class - 1) + random_start + r - 1 end

    elseif dataset == "Faces"
        class == 0 ? begin start_index = 1; end_index = 950 end : begin start_index = 1001; end_index = 1950 end 
        ret_index = rand(start_index:end_index, repeats)

    elseif dataset == "Symbols"
        ret_index = rand(1:1000, repeats)

    elseif dataset == "SymbolsFacesCombined"
        ret_index = rand(1:3000, repeats)

    elseif dataset == "Sequences"
        ret_index = rand(1:18596, repeats)
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

