module Sampling
    ## ANGULAR DISTRIBUTIONS
    using Distributions
    using Random
    # Random.default_rng()
    
    function polar_angle(dist::String, size; fixed_value=0, rng=TaskLocalRNG())
        dist = uppercase(dist)
        if size > 1
            if dist == "HS"
                return 2*asin.(rand(rng,size).^(1/4))
            elseif dist == "IHS"
                return 2*asin.(sqrt.(1 .- sqrt.(1 .- rand(rng,size))))
            elseif dist == "AVG"
                return 5*pi/8 * ones(size)
            elseif dist == "ISO"
                return 2*asin.(sqrt.(rand(rng,size)))
            elseif dist == "CONSTANT"
                return fill(fixed_value,size)
            elseif dist == "CQDRABI"
                ζ = rand(rng,size)
                return 2 .* asin.(sqrt.(1 .- ζ.^(1/4)))
            elseif dist == "CQDRABI2"
                ζ = rand(rng,size)
                return 2 .* asin.((1 .- ζ.^(1/4)).^2)
            else
                return print("Choose a valid distribution for the polar angle")
            end
        else
            if dist == "HS"
                return 2*asin(rand(rng)^(1/4))
            elseif dist == "IHS"
                return 2*asin(sqrt(1-sqrt(1-rand(rng))))
            elseif dist == "AVG"
                return 5*pi/8 
            elseif dist == "ISO"
                return 2*asin(sqrt(rand(rng)))
            elseif dist == "CONSTANT"
                return fixed_value
            elseif dist == "CQDRABI"
                ζ = rand(rng)
                return 2 * atan(sqrt(1 - (1-ζ)^(1/4)), (1-ζ)^(1/8))
            elseif dist == "CQDRABI2"
                ζ = rand(rng)
                return 2 * asin((1 - (1 - ζ)^(1/4))^2)
            else
                return print("Choose a valid distribution for the polar angle")
            end
        end

    end

    function azimu_angle(dist::String, size; fixed_value = 0, rng=TaskLocalRNG())
        dist = uppercase(dist)
        if size >1
            if dist == "ISO"
                return 2*pi*rand(rng,size)
            elseif dist == "CONSTANT"
                return fill(fixed_value,size)
            else
                return print("Choose a valid distribution for the azimuthal angle")
            end
        else
            if dist == "ISO"
                return 2*pi*rand(rng)
            elseif dist == "CONSTANT"
                return fixed_value
            else
                return print("Choose a valid distribution for the azimuthal angle")
            end
        end
    end

    function rotated_hs(θᵣ,ϕᵣ,size; rng=TaskLocalRNG()) 
        # θᵣ rotation angle around the y axis : first 
        # ϕᵣ rotation angle around the z axis : second
        
        θ_hs = 2*asin.(rand(rng,size).^(1/4))
        ϕ_hs   = 2*pi*rand(rng, size)
    
        x_r = cos(ϕᵣ) .* cos(θᵣ) .* sin.(θ_hs) .* cos.(ϕ_hs) .- sin(ϕᵣ) .* sin.(θ_hs) .* sin.(ϕ_hs) .+ cos(ϕᵣ) .* sin(θᵣ) .* cos.(θ_hs)
        y_r = sin(ϕᵣ) .* cos(θᵣ) .* sin.(θ_hs) .* cos.(ϕ_hs) .+ cos(ϕᵣ) .* sin.(θ_hs) .* sin.(ϕ_hs) .+ sin(ϕᵣ) .* sin(θᵣ) .* cos.(θ_hs)
        z_r = -sin(θᵣ) .* sin.(θ_hs) .* cos.(ϕ_hs) .+ cos(θᵣ) .* cos.(θ_hs)
    
        θ_R = acos.(z_r)
        ϕ_R   = pi .+ sign.(y_r) .* (-pi .+ acos.(x_r ./ sqrt.(1 .- z_r.^2) ))
    
        return θ_R, ϕ_R
    end

    function PDF_θϕ(pdf::String, θ, ϕ ; θo=0, Ry=0, Rz=0, )
        pdf = uppercase(pdf)
        if pdf == "ISO"
            1/4π
        elseif pdf == "HS"
            1/2π .* sin.(θ/2).^2
        elseif pdf == "IHS"
            1/2π .* cos.(θ/2).^2
        elseif pdf == "ROTATED"
            1/4π * ( 1 .- cos(Ry).*cos.(θ) - sin(Ry).*sin.(θ).*cos.(ϕ .- Rz))
        elseif pdf == "CQDRABI"
            1/π * cos.(θ/2).^6
        elseif pdf == "CONSTANT"
            ϵ = 0.01
            1/2π * 1 ./ sin.(θ) .* 1 / (sqrt(2π)*ϵ ) .* exp.(-1/(2*ϵ^2)*(θ .- θo).^2)
        else
            print("Choose among the valid Probability Density Functions")
        end
    end

    function PDF_θ(pdf::String, θ ;  θo=0, Ry=0 , Rz=0, )
        pdf = uppercase(pdf)
        if pdf == "ISO"
            1/2 .* sin.(θ)
        elseif pdf == "HS"
            sin.(θ/2).^2 .* sin.(θ)
        elseif pdf == "IHS"
            cos.(θ/2).^2 .* sin.(θ)
        elseif pdf == "ROTATED"
            1/2 * ( 1 .- cos(Ry).*cos.(θ)) .* sin.(θ)
        elseif pdf == "CQDRABI"
            2 * cos.(θ/2).^6 .* sin.(θ)
        elseif pdf == "CONSTANT"
            ϵ = 0.01
            1/3*1 ./ sin.(θ) .* 1 / (sqrt(2π)*ϵ ) .* exp.(-1/(2*ϵ^2)*(θ .- θo).^2)
        else
            print("Choose among the valid Probability Density Functions")
        end
    end

    function PDF_ϕ(pdf::String, ϕ ; Ry=0 , Rz=0)
        pdf = uppercase(pdf)
        if pdf == "ROTATED"
            1/8π * ( 4 .- π * cos.(ϕ .- Rz) .* sin.(Ry) )
        else 
            1/2π * ones(length(ϕ))
        end
    end


    function PDF_hs(size)
        theta_hs = 2*asin.(rand(size).^(1/4))
        phi_hs   = 2*pi*rand(size)
        return theta_hs, phi_hs
    end


    function constrained_normal(interval::Vector{Float64}, mean, variance, num_samples; rng=TaskLocalRNG())
        # Set the seed for the random number generator
        Random.seed!(rng)
        
        # Calculate the standard deviation from the variance
        std_dev = sqrt(variance)

        # Create a truncated normal distribution with the given mean and standard deviation
        dist = Truncated(Normal(mean, std_dev), interval[1], interval[2])
     
        random_numbers = [rand(rng, dist) for _ in 1:num_samples]
        return rand(dist,num_samples)
    end


    function truncated_normal_in_range(mean, variance, num_samples)
        """
        truncated_normal_in_range(mean, variance, num_samples)

        Generate random numbers from a normal distribution with a given mean and variance, truncated to be within the [0, π] range.

        # Arguments
        - `mean::Number`: The desired mean of the distribution.
        - `variance::Number`: The desired variance of the distribution.
        - `num_samples::Integer`: The number of random samples to generate.

        # Returns
        An array of `num_samples` random numbers from the normal distribution, truncated to be within the [0, π] range.
        """
        # Generate random numbers from a standard normal distribution
        random_numbers = randn(num_samples)
        
        # Scale the random numbers to have the desired variance
        scaled_numbers = sqrt(variance) * random_numbers
        
        # Shift the scaled numbers to have the desired mean
        shifted_numbers = scaled_numbers .+ mean

        # Wrap numbers around [0, 2π] and then truncate them to [0, π]
        shifted_numbers = π .- abs.( mod.(shifted_numbers, 2π) .- π )
        
        # Truncate numbers to be within the [0, π] range
        truncated_numbers = max.(0.0, min.(π, shifted_numbers))
        
        return truncated_numbers
    end


    function rotation_y(x)
        """
        Generate a 3x3 rotation matrix for a rotation around the y-axis by an angle `x`.

        # Arguments
        - `x::Real`: The angle of rotation around the y-axis, in radians.

        # Returns
        - `Array{Float64,2}`: A 3x3 rotation matrix representing the rotation.
        """
        return [ cos(x) 0 sin(x) ; 0 1 0 ; -sin(x) 0 cos(x)]
    end

    function rotation_z(x)
        """
        Generate a 3x3 rotation matrix for a rotation around the z-axis by an angle `x`.

        # Arguments
        - `x::Real`: The angle of rotation around the z-axis, in radians.

        # Returns
        - `Array{Float64,2}`: A 3x3 rotation matrix representing the rotation.
        """
        return [ cos(x) -sin(x) 0 ; sin(x) cos(x) 0 ; 0 0 1]
    end

end


#########################
module PolarFunctions#theta_funcs
    function theta_safe(x, tiny_angle)
        theta_m = mod(x,float(pi))
        if theta_m < tiny_angle
            ts = tiny_angle
        elseif theta_m > pi - tiny_angle
            ts = - tiny_angle
        else
            ts = x
        end
        return ts
    end

    function theta_wrap(x)
        pi .- abs.( mod.(x,2*pi) .- pi)
    end
end


########################
module Branching
    function branching_cqd(x,y)
        # x : θ_nucleus
        # y : θ_electron
        if isa(x, Number) && isa(y,Number)
            return isnan(x) || isnan(y) ? NaN : (x < y ? 1 : 0)
        elseif isa(x, AbstractMatrix) && isa(y, AbstractMatrix)
            result = similar(x)
            for i in eachindex(x)
                result[i] = isnan(x[i]) || isnan(y[i]) ? NaN : (x[i] < y[i] ? 1 : 0)
            end
            return result
        else
            error("x and y must be either both matrices or both floats.")
        end
    end

    # Avoid NaN in the probability of spin flip calculation
    import Statistics: mean
    nanmean(x) = mean(filter(!isnan, x))
    function nanmean(x, y)
        return mapslices(nanmean, x, dims=y)
    end

end

# Plotting tools
module PlottingTools
    using LaTeXStrings
    function pitick(start, stop, denom; mode=:text)
        a = Int(cld(start, π/denom))
        b = Int(fld(stop, π/denom))
        tick = range(a*π/denom, b*π/denom; step=π/denom)
        ticklabel = piticklabel.((a:b) .// denom, Val(mode))
        tick, ticklabel
    end

    function piticklabel(x::Rational, ::Val{:text})
        iszero(x) && return "0"
        S = x < 0 ? "-" : ""
        n, d = abs(numerator(x)), denominator(x)
        N = n == 1 ? "" : repr(n)
        d == 1 && return S * N * "π"
        S * N * "π/" * repr(d)
    end

    function piticklabel(x::Rational, ::Val{:latex})
        iszero(x) && return L"0"
        S = x < 0 ? "-" : ""
        n, d = abs(numerator(x)), denominator(x)
        N = n == 1 ? "" : repr(n)
        d == 1 && return L"%$S%$N\pi"
        L"%$S\frac{%$N\pi}{%$d}"
    end

end


# Special functions
module SpecialFunctions0
    function dirac_delta(x,x0,ϵ) # https://functions.wolfram.com/GeneralizedFunctions/DiracDelta/02/
        1/π * ϵ ./ ( (x .- x0 ).^2 .+ ϵ^2)
    end

end