using LinearAlgebra

function print_descent(W::Array{<:Real, 2}, x::Vector{<:Real}, y::Vector{<:Real})
    y0 = W'*x
    error_start = round(norm(W*y0 - x), sigdigits = 5)
    error_end = round(norm(W*y - x), sigdigits = 5)

    l0_start = sum(y0 .> 0.0)
    l0_end = sum(y .> 0.0)

    print("Error start : ", error_start, ", Error end : ", error_end, "\n")
    print("L0 start : ", l0_start, ", L0 end : ", l0_end, "\n")
    print("\n")

    return nothing
end


function calculate_gain(A::AbstractArray{<:Real, 2})
    b = rand(size(A, 2))
    col_normalize!(b, b, showNorms = false)
    
    for i = 1:50 col_normalize!(b, A*b, showNorms = false) end
    
    return b'*A*b
end


@inline function threshold!(dest::Vector{Float64}, x::Vector{Float64}, lambda::Float64)
    for i in eachindex(x)
        abs(x[i]) > lambda ? dest[i] = sign(x[i])*(abs(x[i]) - lambda) : dest[i] = 0.0
    end

    return nothing
end

@inline function relu!(dest::Array{Float64}, y::Array{Float64}; threshold::Float64 = 0.0)
    if isa(y, Vector)
        for i in eachindex(y) y[i] < threshold ? dest[i] = 0.0 : dest[i] = y[i] end

    elseif isa(y, Matrix)
        m, n = size(y)
        for j = 1:n
            for i = 1:m
                y[i, j] < threshold ? dest[i, j] = 0.0 : dest[i, j] = y[i, j]
            end
        end
    end

    return nothing
end


function update_y(W::AbstractArray{<:Real, 2}, x::Vector{<:Real}, usr_args::Dict{Symbol, Any})
    lambda = get(usr_args, :lambda, 0.01)
    verbose = get(usr_args, :verbose, false)
    w_lateral = get(usr_args, :w_lateral, W'*W)

    input = W'*x
    
    gain = 10*calculate_gain(w_lateral)
    gain_inv = inv(gain)

    y_prev = copy(input)
    y = copy(y_prev)
    
    for iter = 1:500
        iter < 3 ? l = 0.0 : l = lambda
        
        @inbounds y .= y_prev .+ gain_inv .* (input .- w_lateral*y_prev)
        threshold!(y, y, l*gain_inv) 
        relu!(y, y)
        copyto!(y_prev, y)
        
    end
    verbose ? print_descent(W, x, y) : nothing
    
    return y
end


function update_phi!(W::AbstractArray{<:Real, 2}, x::Vector{<:Real}, y::Vector{<:Real}, usr_args::Dict{Symbol, Any})
    step_size = get(usr_args, :step_size, 0.000005)
    n = 1000
    norm_y = norm(y)

    if !iszero(norm_y)
        hebbian_coeff = ((1 - (1 - step_size)^n)/norm_y)*(1 - (1/norm_y))
        anti_hebbian_coeff = ((1 - (1 - step_size)^n)/norm_y^2)

        delta = W*y - x
        delta_W = hebbian_coeff*x*y' - anti_hebbian_coeff*delta*y'
        broadcast!(+, W, W, delta_W)

    end
  
    return nothing
end
        

function simulate_layer(W0::AbstractArray{<:Real, 2}, x::Vector{<:Real}; kwargs...)
    usr_args = Dict{Symbol, Any}(kwargs)
    mode = get(usr_args, :mode, 0)

    if mode == 0
        y = update_y(W0, x, usr_args)
        return y
        
    elseif mode == 1
        W = copy(W0)
        y = update_y(W, x, usr_args)
        update_phi!(W, x, y, usr_args)

        return W, y

    else
        error("improper mode")
    end

end
