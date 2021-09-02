using LinearAlgebra
using FileIO, JLD

function fetch_inputs(usr_args::Dict{Symbol, Any})
    test_set_file = string(pwd(), "/data-files/test_set")
    input_matrix = initialize_input_matrix(usr_args[:dataset], length(usr_args[:input_indices]))

    input_count = 1
    for inputNo in usr_args[:input_indices]
        print_progress("fetching representation inputs...", input_count, length(usr_args[:input_indices]))
        input_matrix[:, input_count] = get_input(usr_args[:dataset], inputNo)
        input_count += 1

    end
    
    save(File{format"JLD"}(test_set_file), "test_set", input_matrix)
    return nothing
end

load_test_set() = load_jld_data("test_set", "test_set")
remove_test_set() = rm(string(pwd(), "/data-files/test_set"))


function check_file(file::String)
    seq = Array{String, 1}()
    if isfile(file)
        f = jldopen(file)
        try
            seq = names(f)
        catch
            nothing
        finally
            close(f)
        end

        return seq
    else
        error("file id not available")
    end
end

function get_state_sequence(sfid::String)
    dest_dir = string(pwd(), "/data-files/state-files/")
    c_file = string(dest_dir, "States-c-", sfid)
    d_file = string(dest_dir, "States-d-", sfid)

    seq_c = check_file(c_file)
    seq_d = check_file(d_file)

    if !isempty(seq_c) && !isempty(seq_d) && seq_c == seq_d
        return seq_d
    else
        error("sequence mismatch")
    end
end

function representation_loop!(seq_id::String, usr_args::Dict{Symbol, Any}, connections...)
    Wd = connections[1]
    w_lateral = connections[2]
    init_proj = connections[3]
    test_inputs = load_test_set()

    nlayers = length(init_proj)
    number_of_inputs = size(test_inputs, 2)
    
    representation_c, representation_d = initialize_representation(number_of_inputs, init_proj)

    for input_count in 1:number_of_inputs
        input = test_inputs[:, input_count]

        for layer in 1:nlayers
            yd = simulate_layer(Wd[layer], input, mode = usr_args[:mode], lambda = usr_args[:lambda], verbose = usr_args[:verbose])
            Wc = get_selforg_dictionary(Wd[layer], init_proj[layer])
            yc = simulate_layer(Wc, input, w_lateral = w_lateral[layer], mode = usr_args[:mode], lambda = usr_args[:lambda], verbose = usr_args[:verbose])

            representation_c[layer][:, input_count] = yc
            representation_d[layer][:, input_count] = yd

            input = modulate(yc, yd, modulation_factor = usr_args[:modulation_factor])
        end
        print_progress(string("representing data for state ", seq_id, " of ", usr_args[:sequence][end], "..."), input_count, number_of_inputs)

    end
    save_state(representation_c, seq_id, "rc")
    save_state(representation_d, seq_id, "rd")

    return nothing
end

function run_representation_procedure(usr_args::Dict{Symbol, Any})
    get!(usr_args, :lambda, 0.1)
    get!(usr_args, :mode, 0)
    get!(usr_args, :verbose, false)
    get!(usr_args, :modulation_factor, 0.5)

    sfid = usr_args[:sfid]

    init_proj = load_file(sfid, "0", "d")

    for seq_id in usr_args[:sequence]
        if seq_id != "0"
            W = load_file(sfid, seq_id, "d")
            w_lateral = load_file(sfid, seq_id, "c")
            representation_loop!(seq_id, usr_args, W, w_lateral, init_proj)
        end

    end
    remove_test_set()
    save_state_sequences()
    save_parameters(usr_args)

    return nothing
end



function represent_data(dataset::String, input_indices::Union{Int64, UnitRange{Int64}, Vector{Int64}}, sfid::String; kwargs...)
    usr_args = Dict{Symbol, Any}(kwargs)
    
    usr_args[:dataset] = dataset

    isa(input_indices, Int64) ? begin usr_args[:input_indices] = 1:input_indices end : usr_args[:input_indices] = input_indices
    fetch_inputs(usr_args)
    
    usr_args[:sequence] = get_state_sequence(sfid)
    usr_args[:sfid] = sfid

    run_representation_procedure(usr_args)

    return nothing

end