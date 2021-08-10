using LinearAlgebra, FileIO, JLD

function load_jld_data(filename::String, varname::String)
    isdir("data-files") ? fname = string(pwd(), "/data-files/", filename) : error("data-files not in working directory")
    
    if isfile(fname)
        return load(File{format"JLD"}(fname), varname)
    else
        error("file not in data-files")
    end
end


function col_normalize(x::AbstractArray{<:Real}; showNorms::Bool = false)
    if isa(x, Union{Matrix, LinearAlgebra.Adjoint, SubArray{eltype(x), 2}})
        m, n = size(x)
    elseif isa(x, Union{Vector, SubArray{eltype(x), 1}})
        m = length(x)
        n = 1
    end

    y = Array{Float64, 2}(undef, m, n)
    col_norms = Array{Float64, 1}(undef, n)

    for i in 1:n
        col = @view x[:, i]
        @inbounds col_norms[i] = norm(col)
        @inbounds col_norms[i] != 0.0 ? y[:, i] .= col./col_norms[i] : y[:, i] .= 0.0
    end

    if showNorms 
        return y, col_norms
    else
        return y
    end
    
end


function col_normalize!(dest::AbstractArray{Float64}, x::AbstractArray{<:Real}; showNorms::Bool = false)
    if isa(x, Union{Matrix, LinearAlgebra.Adjoint, SubArray{eltype(x), 2}})
        n = size(x, 2)
    elseif isa(x, Union{Vector, SubArray{eltype(x), 1}})
        n = 1
    end

    col_norms = Array{Float64, 1}(undef, n)

    for i in 1:n
        col = @view x[:, i]
        @inbounds col_norms[i] = norm(col)
        @inbounds col_norms[i] != 0.0 ? dest[:, i] .= col./col_norms[i] : dest[:, i] .= 0.0
    end

    if showNorms
        return col_norms
    end
    
end

function print_progress(mssg::String, current::Int64, termination::Int64)
    p = 100*current/termination
    p < 1.0 ? begin print("\r"); print(mssg, " |", " "^100, "| ") end : nothing

    try
        n = convert(Int64, p)
        print("\r")
        if n < 100 
            print(mssg, " |", "\u220e"^n, " "^(100 - n), "| ", string(n), "\uff05")
        else
            print(mssg, " |", "\u220e"^n, " "^(100 - n), "| ", string(n), "\uff05\n")
        end
    catch
        nothing
    end
end