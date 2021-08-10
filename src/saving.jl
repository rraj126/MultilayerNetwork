using JLD, FileIO, Dates

function date_stamp()
    t = Dates.now()
    return Dates.format(t, "-ud-HHMM")
end

@inline function check_identifier(identifier::String)
    if identifier in ["c", "d", "rc", "rd", "p"]
        return nothing
    else
        error("wrong identifier")
    end
end 


function save_parameters(Args::Dict{Symbol, Any})
    param_dir = string(pwd(), "/parameter-files/")
    !isdir(param_dir) ? mkdir(param_dir) : nothing

    t = date_stamp()
    filename = string(param_dir, "parameters", t, ".txt")

    print("saving parameter file in ", param_dir, "... \n")
    f = open(filename, "a")
    for k in keys(Args) write(f, string(k, " : ", string(Args[k]), "\n")) end
    close(f)

    return nothing
end


function save_state(state::Array{Array{Float64, 2}, 1}, varname::String, identifier::String)
    check_identifier(identifier)
    data_dir = string(pwd(), "/data-files/")
    !isdir(data_dir) ? mkdir(data_dir) : nothing

    if identifier == "p"
        dest_dir = string(data_dir, "projection-files/")
    elseif occursin("r", identifier)
        dest_dir = string(data_dir, "representation-files/")
    else
        dest_dir = string(data_dir, "state-files/")
    end
    !isdir(dest_dir) ? mkdir(dest_dir) : nothing

    filename = string(dest_dir, "state-file-", identifier)
    isfile(filename) ? f = jldopen(filename, "r+") : f = jldopen(filename, "w")

    try
        write(f, varname, state)
    finally
        close(f)
    end

    return nothing
end


function save_state_sequences()
    data_dir = string(pwd(), "/data-files/")
    t = date_stamp()

    for identifier in ["c", "d", "rc", "rd", "p"]
        if identifier == "p"
            dest_dir = string(data_dir, "projection-files/")
    
            src = string(dest_dir, "state-file-", identifier)
            dest = string(dest_dir, "Projection", t)
    
            isfile(src) ? begin print("saving projection file in ", dest_dir, "... \n"); mv(src, dest) end : nothing

        elseif occursin("r", identifier)
            dest_dir = string(data_dir, "representation-files/")

            src = string(dest_dir, "state-file-", identifier)
            dest = string(dest_dir, "Representations-", identifier[2], t)

            isfile(src) ? begin print("saving representation file in ", dest_dir, "... \n"); mv(src, dest) end : nothing

        else
            dest_dir = string(data_dir, "state-files/")

            src = string(dest_dir, "state-file-", identifier)
            dest = string(dest_dir, "States-", identifier, t)

            isfile(src) ? begin print("saving network-state file in ", dest_dir, "... \n"); mv(src, dest) end : nothing

        end
    end

    return nothing
end


function load_file(fid::String, varname::String, identifier::String)
    check_identifier(identifier)
    data_dir = string(pwd(), "/data-files/")
    
    if isdir(data_dir)
        if identifier == "p"
            dest_dir = string(data_dir, "projection-files/")
            isempty(fid) ? filename = string(dest_dir, "state-file-", identifier) : filename = string(dest_dir, "Projection-", fid)

        elseif occursin("r", identifier)
            dest_dir = string(data_dir, "representation-files/")
            isempty(fid) ? filename = string(dest_dir, "state-file-", identifier) : filename = string(dest_dir, "Representations-", identifier[2], "-", fid)

        else
            dest_dir = string(data_dir, "state-files/")
            isempty(fid) ? filename = string(dest_dir, "state-file-", identifier) : filename = string(dest_dir, "States-", identifier, "-", fid)

        end

        isdir(dest_dir) && isfile(filename) ? y = load(File{format"JLD"}(filename), varname) : error("file not in data-files")

    else
        error("data-files not in pwd")
    end

    return y
end


load_file(varname::String, identifier::String) = load_file("", varname, identifier)