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

    filename = "T_tilde_$tracers/T_tilde_l_$l_string.npy"
    #filename = "test/T_tilde_l_$l_string.npy"

    if isfile(filename)
        println("File $filename already exists. Moving on."); flush(stdout)
    else
        z_b = npzread("background/z.npy")
        χ = npzread("background/chi.npy")
        z_of_χ = DataInterpolations.AkimaInterpolation(z_b, χ);
        pk_dict = npzread("../N5K/input/pk.npz")
        Pklin = pk_dict["pk_lin"]
        k = pk_dict["k"]
        z = pk_dict["z"];
        y = LinRange(log10(first(k)),log10(last(k)), length(k))
        x = LinRange(first(z), last(z),length(z))
        InterpPmm = Interpolations.interpolate(log10.(Pklin),BSpline(Cubic(Line(OnGrid()))))
        InterpPmm = scale(InterpPmm, x, y)
        InterpPmm = Interpolations.extrapolate(InterpPmm, Line());
        power_spectrum(k, χ1, χ2) = @. sqrt(10^InterpPmm(z_of_χ(χ1),log10(k)) * 10^InterpPmm(z_of_χ(χ2),log10(k)));

        #Define our grid
        kmax = 200/13 
        kmin = 2.5/7000
        nχ = 100
        #NR = 300
        R = unique(vcat(LinRange(0,0.9,300), LinRange(0.9,1,151)))
    

        #@time T = Will.turbo_T̃(power_spectrum, ℓ, χ, kmin, kmax, tracers)
        #@time T = Will.turbo_T̃_Rχ(power_spectrum, ℓ, nχ, NR, kmin, kmax, tracers)
        @time T = Will.turbo_T̃_Rχ_grid(power_spectrum, ℓ, nχ, R, kmin, kmax, tracers)

        npzwrite(filename, T)
    end

end


main()
