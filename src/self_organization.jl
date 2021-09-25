using LinearAlgebra, Statistics


function update_SynPot!(SynPot::Dict{Tuple{Int64, Int64}, Int64}, y::BitArray{1})
    for i in 1:length(y)-1
         @inbounds if y[i]
            for j in i+1:length(y) 
                @inbounds if y[j]
                    haskey(SynPot, (i, j)) ? SynPot[i, j] += 1 : SynPot[i, j] = 1
                else
                    haskey(SynPot, (i, j)) ? SynPot[i, j] -= 1 : SynPot[i, j] = -1
                end
            end
        else
            for j in findall(@view y[i+1:end])
                haskey(SynPot, (i, j+i)) ? SynPot[i, j+i] -= 1 : SynPot[i, j+i] = -1
            end
        end
    end

    return nothing
end


function get_selforg_dictionary(W::Matrix{<:Real}, init_proj::Matrix{<:Real})
    d = similar(W)
    copyto!(d, W)
    d .= d .- init_proj
    iszero(std(d)) ? d .= 0.0 : d .= d./std(d)
    d .= d.*(d .> 1.0)
    
    col_normalize!(d, d, showNorms = false)
    return d
end


function self_organize_layer(W::Matrix{<:Real}, init_proj::Matrix{<:Real}, input::Vector{<:Real}; kwargs...)
    haskey(kwargs, :SynPot) ? SynPot = kwargs[:SynPot] : SynPot = Dict{Tuple{Int64, Int64}, Int64}()
    
    selforg_dictionary = get_selforg_dictionary(W, init_proj)
    
    if haskey(kwargs, :w_lateral)
        if haskey(kwargs, :lambda)
            y = simulate_layer(selforg_dictionary, input, mode = 0, w_lateral = kwargs[:w_lateral], lambda = kwargs[:lambda])
            response = ones(size(y)) .* (y .> 0)
            update_SynPot!(SynPot, y .> 0)

        else
            y = simulate_layer(selforg_dictionary, input, mode = 0, w_lateral = kwargs[:w_lateral])
            response = ones(size(y)) .* (y .> 0)
            update_SynPot!(SynPot, y .> 0)

        end

    else
        y = transpose(selforg_dictionary)*input
        y .= y ./ maximum(y)
        response = ones(size(y)) .* (y .> 0.8)
        update_SynPot!(SynPot, y .> 0.8)

    end

    return response

end