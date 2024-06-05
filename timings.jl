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
        "Threads"
            help = "Number of threads to employ"
            arg_type = Int32 
            required = true
    end

    return parse_args(s)
end

function C_ell_computation_tullio(w, WA, WB)
    @tullio Cℓ[i,j,k] := w[i,l,m]*WA[j,l]*WB[k,m]
end

function main()

    args = parse_commandline()
    t = args["Threads"]
    println("Using $t threads."); flush(stdout)

    settings = "conservative"

    if settings == "conservative"
        n_k = 96
        n_z1 = 104
        n_z2 = 240
        n_z1_z2 = n_z1*n_z2#Int(floor(n_z1*n_z2/2))
        n_l = 18
        n_tomo1 = 10
        n_tomo2 = 10
        @info "Conservative settings"
    elseif settings == "optimistic"
        n_k = 80
        n_z1 = 104
        n_z2 = 204
        n_z1_z2 = n_z1*n_z2#Int(floor(n_z1*n_z2/2))
        n_l = 16
        n_tomo1 = 10
        n_tomo2 = 10
        @info "Optimistic settings"
    else
        error("Not valid settings!")
    end

    mytype = Float32
    x = rand(mytype, n_z1_z2, n_k)
    FFTW.set_num_threads(t)
    y = FFTW.plan_r2r(x, FFTW.DHT, [2])

    println("Benchmarking the FFT..."); flush(stdout)
    a1 = @benchmark $y*$x
    a2 = @benchmark $y*$x
    a3 = @benchmark $y*$x
    a4 = @benchmark $y*$x
    a5 = @benchmark $y*$x
    a6 = @benchmark $y*$x
    a7 = @benchmark $y*$x
    a8 = @benchmark $y*$x
    a9 = @benchmark $y*$x
    a10 = @benchmark $y*$x

    benchmark1 = [mean(a1.times), mean(a2.times), mean(a3.times), mean(a4.times), mean(a5.times), mean(a6.times), mean(a7.times), mean(a8.times), mean(a9.times), mean(a10.times)]

    coeff = rand(mytype, n_z1, n_z2, n_k)
    T = rand(mytype, n_l, n_z1, n_z2, n_k)
    w = Will.w_ell_tullio(coeff, T)

    println("Benchmarking w computation..."); flush(stdout)
    #println(coeff)
    b1 = @benchmark Will.w_ell_tullio($coeff, $T)
    b2 = @benchmark Will.w_ell_tullio($coeff, $T)
    b3 = @benchmark Will.w_ell_tullio($coeff, $T)
    b4 = @benchmark Will.w_ell_tullio($coeff, $T)
    b5 = @benchmark Will.w_ell_tullio($coeff, $T)
    b6 = @benchmark Will.w_ell_tullio($coeff, $T)
    b7 = @benchmark Will.w_ell_tullio($coeff, $T)
    b8 = @benchmark Will.w_ell_tullio($coeff, $T)
    b9 = @benchmark Will.w_ell_tullio($coeff, $T)
    b10 = @benchmark Will.w_ell_tullio($coeff, $T)

    benchmark2 = [mean(b1.times), mean(b2.times), mean(b3.times), mean(b4.times), mean(b5.times), mean(b6.times), mean(b7.times), mean(b8.times), mean(b9.times), mean(b10.times)]

    WA = rand(mytype, n_tomo1, n_z1)
    WB = rand(mytype, n_tomo2, n_z2)

    Cℓ = C_ell_computation_tullio(w, WA, WB)

    println("Benchmarking the Cl computation..."); flush(stdout)
    c1 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c2 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c3 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c4 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c5 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c6 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c7 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c8 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c9 = @benchmark C_ell_computation_tullio($w, $WA, $WB)
    c10 = @benchmark C_ell_computation_tullio($w, $WA, $WB)

    benchmark3 = [mean(c1.times), mean(c2.times), mean(c3.times), mean(c4.times), mean(c5.times), mean(c6.times), mean(c7.times), mean(c8.times), mean(c9.times), mean(c10.times)]

    total_time = (mean(benchmark1)+3*mean(benchmark2)+3*mean(benchmark3))*1e-9 #seconds

    total_error = (std(benchmark1)+std(benchmark2)+std(benchmark3))*1e-9 #seconds

    open("time_vs_threads.txt", append=true) do file
        write(file, "$t\t$(total_time)\t$(total_error)\n")
    end



end

main()