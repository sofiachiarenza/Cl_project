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
using Interpolations
using FastTransforms
using Revise
using Will

function prepare_interpolator()
    W = npzread("../N5K/input/kernels_fullwidth.npz")
    pk_dict = npzread("../N5K/input/pk.npz")
    # building interpolant chi(z)
    interp = BSplineInterpolation(W["chi_cl"], W["z_cl"], 3, :ArcLen, :Average, extrapolate=true)
    my_chi = interp.(pk_dict["z"])
    my_chi[1] = 0.
    #now, interpolate P(k,chi)
    itp = interpolate((my_chi, log10.(pk_dict["k"])), log10.(pk_dict["pk_lin"]), Gridded(Linear()))
    itp_with_extrapolation = extrapolate(itp, Line());
    return itp_with_extrapolation
end


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
    print("Multipole: ", ℓ, "\n"); flush(stdout)
    l_string = string(round(ell_vector[index]; digits=1))

    tracers = args["Tracers"]

    #filename = "w_brute_$tracers/w_brute_l_$l_string.npy"
    filename = "test/w_brute_l_$l_string.npy"

    if isfile(filename)
        println("File $filename already exists. Moving on."); flush(stdout)
    else
        kmax = 200/13 #N5K challenge values
        kmin = 2.5/7000
        k = LinRange(kmin, kmax, 40000)
        nχ = 100
        χ = LinRange(13, 7000, nχ) 

        function power_spectrum(k, χ1, χ2, interpolator)
            P1 = 10 .^ interpolator(χ1, k)
            P2 = 10 .^ interpolator(χ2, k)
            
            return @. sqrt(P1*P2)
        end

        power_spectrum(k, χ1, χ2) = power_spectrum(k, χ1, χ2, prepare_interpolator())

        result = zeros(nχ, nχ)
        @time Will.brute_w!(result, power_spectrum, ℓ, χ, kmin, kmax, tracers)

        npzwrite(filename, result)
    end


end

main()