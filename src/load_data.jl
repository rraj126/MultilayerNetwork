using MLDatasets, MAT

@inline function load_MNIST_data(input_number::Int64; input_type::DataType = Float64)
    
    return vec(MLDatasets.MNIST.traintensor(input_type, input_number))
    
end

@inline function load_Faces_data(input_number::Int64; input_type::DataType = Float64)
    
    return nothing
    
end

@inline function load_Symbols_data(input_number::Int64, path_to_data::String)
    
    isfile(path_to_data) ? nothing : error("incorrect data path")
    input_number > 1000 ? varname = "SymbolsMatrix" : varname = "CorruptedSymbols"

    file = matopen(path_to_data)
    input = read(file, varname)
    close(file)

    return input[:, input_number]
    
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

    elseif dataset == "Faces"
        sample = load_Faces_data(1)
        max_inputs = 2000
        input_call_function = load_Faces_data

    elseif dataset == "Symbols"
        sample = load_Symbols_data(1, "/Users/rraj/Desktop/Symbols/SymbolsMatrix.mat")
        max_inputs = 1000
        input_call_function = load_Symbols_data

    else
        error("dataset not recognized")

    end
    return sample, max_inputs, input_call_function
end


@inline function get_input_index(class::Union{Int64, Array{Int64, 1}})
    random_start = rand(1:30000)
    ret_index = 0
    class_labels = MNIST.trainlabels(random_start:random_start+500)
    for i in eachindex(class_labels) 
        class_labels[i] in class ? begin ret_index = i+random_start-1; break end : nothing 
    end

    return ret_index
end


@inline function get_input(dataset::String, inputNo::Int64 = 0, input_class::Union{Int64, Array{Int64, 1}} = zeros(Int64, 0))
    _, max_inputs, input_call = get_dataset_sepcifics(dataset)

    if !iszero(inputNo)
        input = input_call(inputNo)
    elseif !isempty(input_class)
        input_index = get_input_index(input_class)
        input = input_call(input_index)
    else
        training_range = convert(Int64, ceil(0.5*max_inputs))
        input = input_call(rand(1:training_range))
    end
    
    return input
end

