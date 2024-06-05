#utils.jl
module Utils

export load_Ts, SimpsonWeightArray, make_grid, grid_interpolator, compute_kernels, C_ell_computation_simpson, C_ell_computation_simpson_uneven_grid, factorial_frac
export cosmic_variance, MAER, power_spectrum_limber, power_spectrum_nl_limber, Cℓ_limber, Cℓ_limber_nl, Σ, Δχ²_tot, Δχ²_vec, Composed_Simpson_Weights

using BenchmarkTools
using LinearAlgebra
using DataInterpolations
using SpecialFunctions
using HCubature
using QuadGK
using Polynomials
using Plots
using FastChebInterp
using ProgressBars
using LaTeXStrings
using Bessels
using Tullio
using FFTW
using LoopVectorization
using NPZ
using Cubature
using FastTransforms
using Interpolations
using Dierckx
using DelimitedFiles

function load_Ts(folder, nχ, nR)
    ell_vector = npzread("ell_vector.npy")[1:21]
    full_T = zeros(21, nχ, nR, 129)
    for i in 1:21
        l_string = string(round(ell_vector[i]; digits=1))
        filename = folder * "/T_tilde_l_$l_string.npy"
        full_T[i,:,:,:] = npzread(filename)
    end
    return full_T
end

function SimpsonWeightArray(n; T=Float64)
    
    number_intervals = floor((n-1)/2)
    weight_array = zeros(n)
    if n == number_intervals*2+1
        for i in 1:number_intervals
            weight_array[Int((i-1)*2+1)] += 1/3
            weight_array[Int((i-1)*2+2)] += 4/3
            weight_array[Int((i-1)*2+3)] += 1/3
        end
    else
        weight_array[1] += 0.5
        weight_array[2] += 0.5
        for i in 1:number_intervals
            weight_array[Int((i-1)*2+1)+1] += 1/3
            weight_array[Int((i-1)*2+2)+1] += 4/3
            weight_array[Int((i-1)*2+3)+1] += 1/3
        end
        weight_array[length(weight_array)]   += 0.5
        weight_array[length(weight_array)-1] += 0.5
        for i in 1:number_intervals
            weight_array[Int((i-1)*2+1)] += 1/3
            weight_array[Int((i-1)*2+2)] += 4/3
            weight_array[Int((i-1)*2+3)] += 1/3
        end
        weight_array ./= 2
    end
    return T.(weight_array)
end

function make_grid(χ, R)
    return vec(χ * R')
end

function grid_interpolator(W, χ, grid)

    W_interp = zeros(length(W[:,1]), length(grid))
    
    for i in 1:length(W[:,1])
        interp = AkimaInterpolation(W[i,:], χ, extrapolate=true)#BSplineInterpolation(W[i,:], χ, 3, :ArcLen, :Average, extrapolate=true)
        W_interp[i,:] = interp.(grid)
    end

    return W_interp
end

function grid_interpolator(W, grid, label::String)
    if label == "C"
        W_array = W["kernels_cl"]
    elseif label == "L"
        W_array = W["kernels_sh"]
    else
        error("Label must be C or L!!!!!!!")
    end

    χ = W["chi_sh"]

    return grid_interpolator(W_array, χ, grid)
end

function compute_kernels(W, χ, R)

    nχ = length(χ)
    nR = length(R)
    
    W_C = reshape(grid_interpolator(W, make_grid(χ, R), "C"), 10, nχ, nR)
    
    χ2_app = zeros(5, nχ*nR)
    for i in 1:5
        χ2_app[i,:] = make_grid(χ, R) .^ 2
    end
    
    W_L = grid_interpolator(W, make_grid(χ, R), "L")
    W_L = reshape( W_L./χ2_app , 5, nχ, nR)

    

    W_C_r1 = W_C[:,:,end]
    W_L_r1 = W_L[:,:,end]

    @tullio K_CC[i,j,c,r] := W_C_r1[i,c] * W_C[j,c,r] + W_C[i,c,r]*W_C_r1[j,c]

    @tullio K_LL[i,j,c,r] := W_L_r1[i,c] * W_L[j,c,r] + W_L[i,c,r]*W_L_r1[j,c]

    @tullio K_CL[i,j,c,r] := W_C_r1[i,c] * W_L[j,c,r] + W_C[i,c,r]*W_L_r1[j,c]

    return K_CC, K_CL, K_LL
end

function C_ell_computation_simpson(w, K) 
    
    nχ = length(w[1,:,1])
    nR = length(w[1,1,:])

    χ = LinRange(26,7000,nχ)
    R = LinRange(0,1, nR+1)[2:end]

    Δχ = ((last(χ)-first(χ))/(nχ-1))
    pesi_χ = SimpsonWeightArray(nχ)
    ΔR = ((last(R)-first(R))/(nR-1))
    pesi_R = SimpsonWeightArray(nR)
        
    @tullio Cℓ[l,i,j] := χ[n]*K[i,j,n,m]*w[l,n,m]*pesi_χ[n]*pesi_R[m]*Δχ*ΔR
    
    return Cℓ
    
end

#=function C_ell_computation_simpson_uneven_grid(w, K, n_a, n_b, α) 
    nχ = length(w[1,:,1])
    nR = n_a + n_b + 2
    T = eltype(w)

    χ = Array(LinRange{T}(26,7000,nχ))
    R1 = Array(LinRange{T}(0,α,n_a + 1))[2:end]
    R2 = Array(LinRange{T}(α,1,n_b + 1))[2:end]

    Δχ = T((last(χ)-first(χ))/(nχ-1))
    pesi_χ = SimpsonWeightArray(nχ, T = T)
    ΔR1 = T((last(R1)-first(R1))/(length(R1)-1))
    pesi_R1 = SimpsonWeightArray(length(R1), T = T)
    ΔR2 = T((last(R2)-first(R2))/(length(R2)-1))
    pesi_R2 = SimpsonWeightArray(length(R2), T = T)

    a = ones(T, length(R1))*ΔR1 
    b = ones(T, length(R2))*ΔR2 
    ΔR = vcat(a,b)
    pesi_R = vcat(pesi_R1,pesi_R2)
        
    @tullio Cℓ[l,i,j] := χ[n]*K[i,j,n,m]*w[l,n,m]*pesi_χ[n]*pesi_R[m]*Δχ*ΔR[m]
    
    return Cℓ
end=#

function Composed_Simpson_Weights(n_a, n_b, ΔxA, ΔxB, T=Float64)
    pesi_a = SimpsonWeightArray(n_a, T = T)*ΔxA
    pesi_b = SimpsonWeightArray(n_b, T = T)*ΔxB

    pesi_tot = vcat(pesi_a[1:end-1],pesi_a[end]+pesi_b[1] ,pesi_b[2:end])
    return pesi_tot
end

function C_ell_computation_simpson_uneven_grid(w, K, χ, n_a, n_b, α) 
    T = eltype(w)
    nχ = length(χ)

    R1 = Array(LinRange{T}(0,α,n_a+1))[2:end]
    R2 = Array(LinRange{T}(α,1,n_b+1))[2:end]

    Δχ = T((last(χ)-first(χ))/(nχ-1))
    pesi_χ = SimpsonWeightArray(nχ, T = T)
    
    ΔR1 = T((last(R1)-first(R1))/(length(R1)-1))
    ΔR2 = T((last(R2)-first(R2))/(length(R2)-1))

    pesi_R = Composed_Simpson_Weights(n_a, n_b+1, ΔR1, ΔR2, T)

        
    @tullio Cℓ[l,i,j] := χ[n]*K[i,j,n,m]*w[l,n,m]*pesi_χ[n]*pesi_R[m]*Δχ
    
    return Cℓ
end


function factorial_frac(n)
    return (n-1)*n*(n+1)*(n+2)
end

function cosmic_variance(Cℓ_CC, Cℓ_LL, Cℓ_CL, ell_vector,f_sky=1)
    n̄_C = 4 * (3437.746771)^2
    n̄_L = 27/5 * (3437.746771)^2
    σ_ϵ = 0.28

    N_CC = 1/n̄_C .* I(10)
    N_LL = σ_ϵ^2/n̄_L .* I(5)

    @tullio σ_CC[l,i,j] := sqrt(((Cℓ_CC[l,i,i]+N_CC[i,i])*(Cℓ_CC[l,j,j]+N_CC[j,j]) + Cℓ_CC[l,i,j]^2)/(f_sky*(2*ell_vector[l]+1)))
    @tullio σ_LL[l,i,j] := sqrt(((Cℓ_LL[l,i,i]+N_LL[i,i])*(Cℓ_LL[l,j,j]+N_LL[j,j]) + Cℓ_LL[l,i,j]^2)/(f_sky*(2*ell_vector[l]+1)))
    @tullio σ_CL[l,i,j] := sqrt(((Cℓ_CC[l,i,i]+N_CC[i,i])*(Cℓ_LL[l,j,j]+N_LL[j,j]) + Cℓ_CL[l,i,j]^2)/(f_sky*(2*ell_vector[l]+1)))

    return σ_CC, σ_LL, σ_CL
end

function MAER(Cℓ_cheb, Cℓ_n5k, σ)
    @tullio maer[l,i,j] := abs(Cℓ_cheb[l,i,j]-Cℓ_n5k[l,i,j])/σ[l,i,j]
    return maer
end

function power_spectrum_limber(ℓ, χ)
    k = (ℓ+0.5) ./ χ
    return @. 10^InterpPmm(z_of_χ(χ),log10(k)) 
end

function power_spectrum_nl_limber(ℓ, χ)
    k = (ℓ+0.5) ./ χ
    return @. 10^InterpPmm_nl(z_of_χ(χ),log10(k)) 
end

function Cℓ_limber_nl(ℓ, χ, tracers)
    n = length(χ)
    W = npzread("../N5K/input/kernels_fullwidth.npz")
    WA = W["kernels_sh"]
    WB = W["kernels_cl"]
    WA_interp = zeros(5,n)
    WB_interp = zeros(10,n)
    
    for i in 1:5
        interp = BSplineInterpolation(WA[i,:], W["chi_sh"], 3, :ArcLen, :Average, extrapolate=true)
        WA_interp[i,:] = interp.(χ)
    end
    
    for i in 1:10
        interp = BSplineInterpolation(WB[i,:], W["chi_cl"], 3, :ArcLen, :Average, extrapolate=true)
        WB_interp[i,:] = interp.(χ)
    end

    if tracers == "CC"
        F = 1
        KA = WB_interp
        KB = WB_interp
    elseif tracers == "CL"
        F = sqrt.(factorial_frac(ℓ))*(ℓ+0.5)^(-2)
        KA = WB_interp
        KB = WA_interp
    elseif tracers == "LL"
        F = factorial_frac(ℓ)*(ℓ+0.5)^(-4)
        KA = WA_interp
        KB = WA_interp
    end

    Δχ = ((χ[n]-χ[1])/(n-1))
    pesi = SimpsonWeightArray(n)

    pk_over_chi = power_spectrum_nl_limber(ℓ, χ) ./ (χ .^ 2)
    
    @tullio Cℓ[i,j] := Δχ*pk_over_chi[m]*KA[i,m]*KB[j,m]*pesi[m]
    return Cℓ
end

function Cℓ_limber(ℓ, χ, tracers)
    n = length(χ)
    W = npzread("../N5K/input/kernels_fullwidth.npz")
    WA = W["kernels_sh"]
    WB = W["kernels_cl"]
    WA_interp = zeros(5,n)
    WB_interp = zeros(10,n)
    
    for i in 1:5
        interp = BSplineInterpolation(WA[i,:], W["chi_sh"], 3, :ArcLen, :Average, extrapolate=true)
        WA_interp[i,:] = interp.(χ)
    end
    
    for i in 1:10
        interp = BSplineInterpolation(WB[i,:], W["chi_cl"], 3, :ArcLen, :Average, extrapolate=true)
        WB_interp[i,:] = interp.(χ)
    end

    if tracers == "CC"
        F = 1
        KA = WB_interp
        KB = WB_interp
    elseif tracers == "CL"
        F = sqrt.(factorial_frac(ℓ))*(ℓ+0.5)^(-2)
        KA = WB_interp
        KB = WA_interp
    elseif tracers == "LL"
        F = factorial_frac(ℓ)*(ℓ+0.5)^(-4)
        KA = WA_interp
        KB = WA_interp
    end

    Δχ = ((χ[n]-χ[1])/(n-1))
    pesi = SimpsonWeightArray(n)

    pk_over_chi = power_spectrum_limber(ℓ, χ) ./ (χ .^ 2)
    
    @tullio Cℓ[i,j] := Δχ*pk_over_chi[m]*KA[i,m]*KB[j,m]*pesi[m]
    return Cℓ
end

function Σ(Cℓ_CC, Cℓ_LL, Cℓ_CL, ℓ,dtype,f_sky=1)
    n̄_C = dtype(4 * (3437.746771)^2)
    n̄_L = dtype(27/5 * (3437.746771)^2)
    σ_ϵ = dtype(0.28)

    N_CC = dtype.(1/n̄_C .* I(10))
    N_LL = dtype.(σ_ϵ^2/n̄_L .* I(5))

    @tullio Σ_CC[l,i,j] := sqrt(2/(f_sky*(2*ℓ[l]+1))) * (Cℓ_CC[l,i,j] + N_CC[i,j])
    @tullio Σ_LL[l,i,j] := sqrt(2/(f_sky*(2*ℓ[l]+1))) * (Cℓ_LL[l,i,j] + N_LL[i,j])
    @tullio Σ_CL[l,i,j] := sqrt(2/(f_sky*(2*ℓ[l]+1))) * Cℓ_CL[l,i,j]

    return dtype.(Σ_CC), dtype.(Σ_LL), dtype.(Σ_CL)
end;

function Δχ²_tot(ΔC, Σ_inv)
    nl = length(Σ_inv[:,1,1])

    dc = 0

    for l in 1:nl
        dc += tr(*(Σ_inv[l,:,:],ΔC[l,:,:])^2)
    end
    return dc 
end

function Δχ²_vec(ΔC, Σ_inv, dtype)
    nl = length(Σ_inv[:,1,1])

    dc = zeros(dtype, nl)

    for l in 1:nl
        dc[l] = tr(*(Σ_inv[l,:,:],ΔC[l,:,:])^2)
    end
    return dc 
end



end