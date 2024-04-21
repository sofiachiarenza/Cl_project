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
using ArgParse
using FastTransforms
using Interpolations
using Will


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "Index"
            help = "ℓ array index"
            arg_type = Int32 
            required = true
        "Tracers"
            help = "Type of tracers: CC, CL, or LL"
            arg_type = String
            required = true
    end

    return parse_args(s)
end


function main()
    args = parse_commandline()
    index = args["Index"]
    ell_vector = npzread("ell_vector.npy")
    ℓ = ell_vector[index]
    print("Multipole: ", ℓ , "\n"); flush(stdout)
    l_string = string(round(ell_vector[index]; digits=1))

    tracers = args["Tracers"]

    #filename = "T_tilde_$tracers/T_tilde_l_$l_string.npy"
    filename = "test/T_tilde_l_$l_string.npy"

    if isfile(filename)
        println("File $filename already exists. Moving on."); flush(stdout)
    else
        #Import window functions
        W = npzread("../N5K/input/kernels_fullwidth.npz")
        WA = W["kernels_sh"]
        WB = W["kernels_cl"]

        nχ = 100
        χ = LinRange(13, 7000, nχ) 
        WA_interp = zeros(5,nχ)
        WB_interp = zeros(10,nχ)

        for i in 1:5
            interp = BSplineInterpolation(WA[i,:], W["chi_sh"], 3, :ArcLen, :Average, extrapolate=true)
            WA_interp[i,:] = interp.(χ)
        end

        for i in 1:10
            interp = BSplineInterpolation(WB[i,:], W["chi_cl"], 3, :ArcLen, :Average, extrapolate=true)
            WB_interp[i,:] = interp.(χ)

            end
        end

        #import power spectrum
        pk_dict = npzread("../N5K/input/pk.npz")

        interp = BSplineInterpolation(W["chi_cl"], W["z_cl"], 3, :ArcLen, :Average, extrapolate=true)
        my_chi = interp.(pk_dict["z"])
        my_chi[1] = 0.
        itp = interpolate((my_chi, log10.(pk_dict["k"])), log10.(pk_dict["pk_lin"]), Gridded(Linear()))
        itp_with_extrapolation = extrapolate(itp, Line())

        #Define our grid
        kmax = 200/13 
        kmin = 2.5/7000
        k = LinRange(kmin, kmax, 40000)
        nχ = 100
        χ = LinRange(13, 7000, nχ)
        n_cheb = 128
        #log_pk_interp = [itp_with_extrapolation(i, j) for i in χ, j in log10.(k)]

        function power_spectrum(k, χ1, χ2, interpolator)
            P1 = 10 .^interpolator(χ1, k)
            P2 = 10 .^interpolator(χ2, k)
            
            return @. sqrt(P1 * P2)
        end

        power_spectrum(k, χ1, χ2) = power_spectrum(k, χ1, χ2, itp_with_extrapolation)

        @time T = Will.turbo_T̃(power_spectrum, ℓ, χ, kmin, kmax, tracers)

        npzwrite(filename, T)
    end


main()