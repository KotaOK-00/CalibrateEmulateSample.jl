# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
ENV["GKSwstype"] = "100"
using Plots
using Random
using JLD2
using Dates
using StatsBase
# CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers

include("GModel.jl")

function main()

    cases = [
        "GP", # diagonalize, train scalar GP, assume diag inputs
    ]

    #### CHOOSE YOUR CASE:
    mask = [1] # 1:8 # e.g. 1:8 or [7]
    for (case) in cases[mask]


        println("case: ", case)
        min_iter = 1
        max_iter = 5 # number of EKP iterations to use data from is at most this

        exp_name = "darcy"
        rng_seed = 940284
        rng = Random.MersenneTwister(rng_seed)

        # loading relevant data
        homedir = pwd()
        println(homedir)
        figure_save_directory = joinpath(homedir, "output/")
        data_save_directory = joinpath(homedir, "output/")
        data_save_file = joinpath(data_save_directory, "calibrate_results.jld2")

        if !isfile(data_save_file)
            throw(
                ErrorException(
                    "data file $data_save_file not found. \n First run: \n > julia --project calibrate.jl \n and store results $data_save_file",
                ),
            )
        end

        ekiobj = load(data_save_file)["eki"]
        eigenfuncs = load(data_save_file)["eigenfuncs"]
        pts_per_dim = load(data_save_file)["pts_per_dim"]
        prior = load(data_save_file)["prior"]
        darcy = load(data_save_file)["darcy"]
        truth_sample = load(data_save_file)["truth_sample"]
        truth_params_constrained = load(data_save_file)["truth_input_constrained"] #true parameters in constrained space
        truth_params = load(data_save_file)["truth_input_unconstrained"] #true parameters in unconstrained space
        Γy = get_obs_noise_cov(ekiobj)


        n_params = length(truth_params) # "input dim"
        output_dim = size(Γy, 1)
        ###
        ###  Emulate: Gaussian Process Regression
        ###

        # Emulate-sample settings
        # choice of machine-learning tool in the emulation stage
        if case == "GP"
            #            gppackage = Emulators.SKLJL()
            gppackage = Emulators.AGPJL()
            mlt = GaussianProcess(gppackage; noise_learn = false)
        end


        # Get training points from the EKP iteration number in the second input term
        N_iter = min(max_iter, length(get_u(ekiobj)) - 1) # number of paired iterations taken from EKP
        min_iter = min(max_iter, max(1, min_iter))
        input_output_pairs = Utilities.get_training_points(ekiobj, min_iter:(N_iter - 1))
        input_output_pairs_test = Utilities.get_training_points(ekiobj, N_iter:(length(get_u(ekiobj)) - 1)) #  "next" iterations
        # Save data
        @save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs

        retained_svd_frac = 1.0
        normalized = true
        # do we want to use SVD to decorrelate outputs
        decorrelate = case ∈ ["RF-vector-nosvd-diag", "RF-vector-nosvd-nondiag"] ? false : true

        emulator = Emulator(
            mlt,
            input_output_pairs;
            obs_noise_cov = Γy,
            normalize_inputs = normalized,
            retained_svd_frac = retained_svd_frac,
            decorrelate = decorrelate,
        )
        optimize_hyperparameters!(emulator)
        # optimize_hyperparameters!(emulator, kernbounds = [fill(-1e2, n_params + 1), fill(1e2, n_params + 1)])

        # Check how well the Gaussian Process regression predicts on the
        # true parameters
        #if retained_svd_frac==1.0
        y_mean, y_var = Emulators.predict(emulator, reshape(truth_params, :, 1), transform_to_real = true)
        y_mean_test, y_var_test =
            Emulators.predict(emulator, get_inputs(input_output_pairs_test), transform_to_real = true)

        println("ML prediction on true parameters: ")
        println(vec(y_mean))
        println("true data: ")
        println(truth_sample) # what was used as truth
        println(" ML predicted standard deviation")
        println(sqrt.(diag(y_var[1], 0)))
        println("ML MSE (truth): ")
        println(mean((truth_sample - vec(y_mean)) .^ 2))
        println("ML MSE (next ensemble): ")
        println(mean((get_outputs(input_output_pairs_test) - y_mean_test) .^ 2))

        #end
        ###
        ###  Sample: Markov Chain Monte Carlo
        ###
        # initial values
        u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
        println("initial parameters: ", u0)





        #=
        # First let's run a short chain to determine a good step size
        mcmc = MCMCWrapper(RWMHSampling(), truth_sample, prior, emulator; init_params = u0)
        new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

        # Now begin the actual MCMC
        println("Begin MCMC - with step size ", new_step)
        T = 100_000
        start_time = now()
        chain = MarkovChainMonteCarlo.sample(mcmc, T; stepsize = new_step, discard_initial = 2_000)
        end_time = now()

        total_time = (end_time - start_time).value
        println("Total MCMC sampling time: ", total_time, " seconds")
        spiter = total_time / T
        println("spiter: ", spiter)
        acc_prob = accept_ratio(chain)
        println("acceptance probability: ", acc_prob)

        #=
        ess_values = MCMCChains.ess(chain)
        println("Effective Sample Size (ESS): ", ess_values)
        println("Min ESS: ", minimum(ess_values))
        println("Median ESS: ", median(ess_values))
        println("Max ESS: ", maximum(ess_values))
        =#

        posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
        post_mean = mean(posterior)
        post_cov = cov(posterior)
        println("post_mean")
        println(post_mean)
        println("post_cov")
        println(post_cov)
        println("D util")
        println(det(inv(post_cov)))
        println(" ")

        param_names = get_name(posterior)
        println("param_names: ", param_names)
        posterior_samples = reduce(vcat, [get_distribution(posterior)[name] for name in get_name(posterior)]) #samples are columns of this matrix
        println(size(posterior_samples))
        n_post = size(posterior_samples, 2)
        plot_sample_id = (n_post - 1000):n_post

        #=
        constrained_posterior_samples = mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)
        println(size(constrained_posterior_samples))
        println("constrained_post_mean: ")
        println(mean(constrained_posterior_samples, dims = 2))
        println("constrained_post_var: ")
        println(var(constrained_posterior_samples, dims = 2))
        println("med: ", median(constrained_posterior_samples, dims = 2))
        println("min: ",  minimum(constrained_posterior_samples, dims = 2))
        println("max: ",  maximum(constrained_posterior_samples, dims = 2))
        constrained_post_κ_field = eigenfuncs * constrained_posterior_samples
        println("size of constrained_post_κ_field: ", size(constrained_post_κ_field))
        println("mean of constrained_post_κ_field: ", mean(constrained_post_κ_field, dims = 2))
        N = Int(sqrt(size(constrained_post_κ_field, 1)))
        println("N: ", N)
        =#

       constrained_posterior_samples =
            transform_unconstrained_to_constrained(prior, posterior_samples[:, plot_sample_id])
        println("size of constrained_posterior_sample: ", size(constrained_posterior_samples))
        println("mean of constrained_posterior_sample: ", mean(constrained_posterior_samples, dims = 2))
        N = Int(sqrt(size(constrained_posterior_samples, 1)))

        #... plot etc
        #=
        n_params = size(posterior_samples, 1)
        p_tp = plot(1:T, posterior_samples[:, 1:2], label="MALA",
        color=colors[1], xlabel="Iteration (t)", ylabel="d_t", title="samples", lw=2)
        figpath = joinpath(figure_save_directory, "posterior_MCMC_" * case * ".pdf")
        savefig(figpath)
        figpath = joinpath(figure_save_directory, "posterior_MCMC_" * case * ".png")
        savefig(figpath)
        =#

        =#





        #=


        ########
        #ESJD analysis
        ########
        function compute_ESJD(sigma_vec::Vector{Float64}, T::Int, n::Int, method)::Matrix{Float64}
            ESJD = Matrix{Float64}(undef, length(sigma_vec), n)
            for (i, sigma) in enumerate(sigma_vec)
                ESJD[i, :] = grad_MCMC_ESJD(T, n, method, sigma)
            end
            return ESJD
        end

        function grad_MCMC_ESJD(T::Int, n::Int, method, sigma::Float64)
            mcmc = MCMCWrapper(method, truth_sample, prior, emulator; init_params = u0)
            chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = sigma, discard_initial = 500)
            posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
            posterior_samples = reduce(vcat, [get_distribution(posterior)[name] for name in get_name(posterior)])
            constrained_posterior = (mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1))'
            n_samples, n_params = size(constrained_posterior)
            esjd = zeros(Float64, n_params)
            for i in 2:n_samples
                esjd = esjd .+ (constrained_posterior[i, :] .- constrained_posterior[i - 1, :]).^ 2 ./ n_samples
            end
            return esjd
        end

        T = 5_000
        n = n_params
        sqrt_n = sqrt(n)
        sigma_vec = 2.38 ./ sqrt_n .* exp.(range(-8, stop = 1, length = 45))
        println("sigma_vec: ", sigma_vec)
        ESJD_MALA = compute_ESJD(sigma_vec, T, n, MALASampling())
        ESJD_RW = compute_ESJD(sigma_vec, T, n, RWMHSampling())
        ESJD_pCN = compute_ESJD(sigma_vec, T, n, pCNMHSampling())
        ESJD_BARKER = compute_ESJD(sigma_vec, T, n, BarkerSampling())
        # ESJD_HMC = compute_ESJD(sigma_vec, T, n, HMCSampling())
        ESJD_infMALA = compute_ESJD(sigma_vec, T, n, infMALASampling())
        ESJD_infmMALA = compute_ESJD(sigma_vec, T, n, infmMALASampling())


        #compute median ESJD OF OTHER COORDINATES
        # Compute max, min, total ESJD of other coordinates
        #compute median ESJD OF OTHER COORDINATES






        # For MALA
        MALA_max = mapslices(maximum, ESJD_MALA[:, 2:end], dims=2)
        MALA_median = mapslices(median, ESJD_MALA[:, 2:end], dims=2)
        MALA_min = mapslices(minimum, ESJD_MALA[:, 2:end], dims=2)
        MALA_total = mapslices(sum, ESJD_MALA[:, 2:end], dims=2)

        # For RWMH
        RW_max = mapslices(maximum, ESJD_RW[:, 2:end], dims=2)
        RW_median = mapslices(median, ESJD_RW[:, 2:end], dims=2)
        RW_min = mapslices(minimum, ESJD_RW[:, 2:end], dims=2)
        RW_total = mapslices(sum, ESJD_RW[:, 2:end], dims=2)

        # For pCN sampler
        pCN_max = mapslices(maximum, ESJD_pCN[:, 2:end], dims=2)
        pCN_median = mapslices(median, ESJD_pCN[:, 2:end], dims=2)
        pCN_min = mapslices(minimum, ESJD_pCN[:, 2:end], dims=2)
        pCN_total = mapslices(sum, ESJD_pCN[:, 2:end], dims=2)

        # For Barker sampler
        BARKER_max = mapslices(maximum, ESJD_BARKER[:, 2:end], dims=2)
        BARKER_median = mapslices(median, ESJD_BARKER[:, 2:end], dims=2)
        BARKER_min = mapslices(minimum, ESJD_BARKER[:, 2:end], dims=2)
        BARKER_total = mapslices(sum, ESJD_BARKER[:, 2:end], dims=2)

        # For infMALA sampler
        infMALA_max = mapslices(maximum, ESJD_infMALA[:, 2:end], dims=2)
        infMALA_median = mapslices(median, ESJD_infMALA[:, 2:end], dims=2)
        infMALA_min = mapslices(minimum, ESJD_infMALA[:, 2:end], dims=2)
        infMALA_total = mapslices(sum, ESJD_infMALA[:, 2:end], dims=2)

        # For infmMALA sampler (if applicable)
        infmMALA_max = mapslices(maximum, ESJD_infmMALA[:, 2:end], dims=2)
        infmMALA_median = mapslices(median, ESJD_infmMALA[:, 2:end], dims=2)
        infmMALA_min = mapslices(minimum, ESJD_infmMALA[:, 2:end], dims=2)
        infmMALA_total = mapslices(sum, ESJD_infmMALA[:, 2:end], dims=2)



        println("maximum value of ESJD_MALA 1st: ", maximum(ESJD_MALA[:, 1]))
        println("at ", argmax(ESJD_MALA[:, 1]))
        println("maximum value of ESJD_RW 1st: ", maximum(ESJD_RW[:, 1]))
        println("at ", argmax(ESJD_RW[:, 1]))
        println("maximum value of ESJD_pCN 1st: ", maximum(ESJD_pCN[:, 1]))
        println("at ", argmax(ESJD_pCN[:, 1]))
        println("maximum value of ESJD_BARKER 1st: ", maximum(ESJD_BARKER[:, 1]))
        println("at ", argmax(ESJD_BARKER[:, 1]))
        println("maximum value of ESJD_infMALA 1st: ", maximum(ESJD_infMALA[:, 1]))
        println("at ", argmax(ESJD_infMALA[:, 1]))
        println("maximum value of ESJD_infmMALA 1st: ", maximum(ESJD_infmMALA[:, 1]))
        println("at ", argmax(ESJD_infmMALA[:, 1]))





        # For MALA sampler
        println("Maximum value of MALA_max ESJD: ", maximum(MALA_max))
        println("At index: ", argmax(MALA_max))
        println("maximum value of median ESJD_MALA: ", maximum(MALA_median))
        println("At index: ", argmax(MALA_median))
       println("Maximum value of MALA_min ESJD: ", maximum(MALA_min))
        println("At index: ", argmax(MALA_min))
        println("Maximum value of MALA_total ESJD: ", maximum(MALA_total))
        println("At index: ", argmax(MALA_total))

        # For Random Walk (RW) sampler
        println("Maximum value of RW_max ESJD: ", maximum(RW_max))
        println("At index: ", argmax(RW_max))
        println("maximum value of median ESJD_RW: ", maximum(RW_median))
        println("At index: ", argmax(RW_median))
        println("Maximum value of RW_min ESJD: ", maximum(RW_min))
        println("At index: ", argmax(RW_min))
        println("Maximum value of RW_total ESJD: ", maximum(RW_total))
        println("At index: ", argmax(RW_total))

        # For pCN sampler
        println("Maximum value of pCN_max ESJD: ", maximum(pCN_max))
        println("At index: ", argmax(pCN_max))
        println("maximum value of median ESJD_pCN: ", maximum(pCN_median))
        println("At index: ", argmax(pCN_median))
        println("Maximum value of pCN_min ESJD: ", maximum(pCN_min))
        println("At index: ", argmax(pCN_min))
        println("Maximum value of pCN_total ESJD: ", maximum(pCN_total))
        println("At index: ", argmax(pCN_total))

        # For Barker sampler
        println("Maximum value of BARKER_max ESJD: ", maximum(BARKER_max))
        println("At index: ", argmax(BARKER_max))
        println("maximum value of median ESJD_BARKER: ", maximum(BARKER_median))
        println("At index: ", argmax(BARKER_median))
        println("Maximum value of BARKER_min ESJD: ", maximum(BARKER_min))
        println("At index: ", argmax(BARKER_min))
        println("Maximum value of BARKER_total ESJD: ", maximum(BARKER_total))
        println("At index: ", argmax(BARKER_total))

        # For infMALA sampler
        println("Maximum value of infMALA_max ESJD: ", maximum(infMALA_max))
        println("At index: ", argmax(infMALA_max))
        println("maximum value of median ESJD_infMALA: ", maximum(infMALA_median))
        println("At index: ", argmax(infMALA_median))
        println("Maximum value of infMALA_min ESJD: ", maximum(infMALA_min))
        println("At index: ", argmax(infMALA_min))
        println("Maximum value of infMALA_total ESJD: ", maximum(infMALA_total))
        println("At index: ", argmax(infMALA_total))

        # For infmMALA sampler
        println("Maximum value of infmMALA_max ESJD: ", maximum(infmMALA_max))
        println("At index: ", argmax(infmMALA_max))
        println("Maximum value of infmMALA_min ESJD: ", maximum(infmMALA_min))
        println("At index: ", argmax(infmMALA_min))
        println("maximum value of median ESJD_infmMALA: ", maximum(infmMALA_median))
        println("at ", argmax(infmMALA_median))
        println("Maximum value of infmMALA_total ESJD: ", maximum(infmMALA_total))
        println("At index: ", argmax(infmMALA_total))








        # PLOT ESJD OF FIRST COORDINATE AND MEDIAN ESJD OF OTHER COORDINATES
        #plot ESJD of first coordinate
        #ylim = (exp(-20), maximum([ESJD_infmMALA[:, 1]; ESJD_infMALA[:, 1]; ESJD_MALA[:, 1]; ESJD_RW[:, 1]; ESJD_pCN[:, 1]; ESJD_BARKER[:, 1]])+0.1)
        ylim = (exp(-15), maximum([ESJD_infMALA[:, 1]; ESJD_MALA[:, 1]; ESJD_RW[:, 1]; ESJD_pCN[:, 1]; ESJD_BARKER[:, 1]]) * 1.1)
        #ylim = (exp(-15), maximum([ESJD_MALA[:, 1]; ESJD_RW[:, 1]; ESJD_pCN[:, 1]; ESJD_BARKER[:, 1]; ESJD_HMC[:, 1]])+0.1)
        # ylim = (exp(-10), maximum([infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total])+0.1)
        ylim = (exp(-15), maximum([infmMALA_total; infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total])+0.1)
        gr(size = (1000, 600))
        p1 = plot(
            sigma_vec,
            ESJD_MALA[:, 1],
            xscale = :log10,
            yscale = :log10,
            label = "MALA",
            color = :red,
            marker = :diamond,
            xlabel = "proposal step-size",
            ylabel = "ESJD",
            ylim = ylim,
            legend = :bottomright,
        )
        plot!(p1, sigma_vec, ESJD_MALA[:, 1], label = nothing, color = :red)
        scatter!(p1, sigma_vec, ESJD_RW[:, 1], label = "RW", color = :black, marker = :circle)
        plot!(p1, sigma_vec, ESJD_RW[:, 1], label = nothing, color = :black)
        scatter!(p1, sigma_vec, ESJD_pCN[:, 1], label = "pCN", color = :green, marker = :cross)
        plot!(p1, sigma_vec, ESJD_pCN[:, 1], label = nothing, color = :green)
        scatter!(p1, sigma_vec, ESJD_BARKER[:, 1], label = "Barker", color = :blue, marker = :utriangle)
        plot!(p1, sigma_vec, ESJD_BARKER[:, 1], label = nothing, color = :blue)
        scatter!(p1, sigma_vec, ESJD_infMALA[:, 1], label = "infMALA", color = :purple, marker = :dtriangle)
        plot!(p1, sigma_vec, ESJD_infMALA[:, 1], label = nothing, color = :purple)
        scatter!(p1, sigma_vec, ESJD_infmMALA[:, 1], label = "infmMALA", color = :orange, marker = :xcross)
        plot!(p1, sigma_vec, ESJD_infmMALA[:, 1], label = nothing, color = :orange)
        title!(p1, "ESJD of coordinate 1")
        display(p1)
        savefig(p1, joinpath(data_save_directory, "Darcy_MCMC_ESJD_1_" * string(n_params) * "_" * string(output_dim) * ".png"))



        #ylim = (exp(-9), maximum([infmMALA_median; infMALA_median; MALA_median; pCN_median; RW_median; BARKER_median])+0.1)
        ylim = (exp(-15), maximum([infMALA_median; MALA_median; pCN_median; RW_median; BARKER_median]) * 1.1)
        gr(size = (1000, 600))
        p2 = plot(
            sigma_vec,
            MALA_median,
            xscale = :log10,
            yscale = :log10,
            label = "MALA",
            color = :red,
            marker = :diamond,
            xlabel = "proposal step-size",
            ylabel = "median ESJD",
            ylim = ylim,
            legend = :bottomright,
        )
        plot!(p2, sigma_vec, MALA_median, label = nothing, color = :red)
        scatter!(p2, sigma_vec, RW_median, label = "RW", color = :black, marker = :circle)
        plot!(p2, sigma_vec, RW_median, label = nothing, color = :black)
        scatter!(p2, sigma_vec, pCN_median, label = "pCN", color = :green, marker = :cross)
        plot!(p2, sigma_vec, pCN_median, label = nothing, color = :green)
        scatter!(p2, sigma_vec, BARKER_median, label = "Barker", color = :blue, marker = :utriangle)
        plot!(p2, sigma_vec, BARKER_median, label = nothing, color = :blue)
        scatter!(p2, sigma_vec, infMALA_median, label = "infMALA", color = :purple, marker = :dtriangle)
        plot!(p2, sigma_vec, infMALA_median, label = nothing, color = :purple)
        scatter!(p2, sigma_vec, infmMALA_median, label = "infmMALA", color = :orange, marker = :xcross)
        plot!(p2, sigma_vec, infmMALA_median, label = nothing, color = :orange)
        title!(p2, "median ESJD of coordinates 2 to last")
        display(p2)
        savefig(p2, joinpath(data_save_directory, "Darcy_MCMC_median_ESJD_" * string(n_params) * "_" * string(output_dim) * ".png"))



        # ylim = (exp(-10), maximum([infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total])+0.1)
        ylim = (exp(-15), maximum([infmMALA_total; infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total]) * 1.1)
        gr(size = (1000, 600))
        p3 = plot(
            sigma_vec,
            MALA_max,
            xscale = :log10,
            yscale = :log10,
            label = "MALA",
            color = :red,
            marker = :diamond,
            xlabel = "proposal step-size",
            ylabel = "max ESJD",
            ylim = ylim,
            legend = :bottomright,
        )
        plot!(p3, sigma_vec, MALA_max, label = nothing, color = :red)
        scatter!(p3, sigma_vec, RW_max, label = "RW", color = :black, marker = :circle)
        plot!(p3, sigma_vec, RW_max, label = nothing, color = :black)
        scatter!(p3, sigma_vec, pCN_max, label = "pCN", color = :green, marker = :cross)
        plot!(p3, sigma_vec, pCN_max, label = nothing, color = :green)
        scatter!(p3, sigma_vec, BARKER_max, label = "Barker", color = :blue, marker = :utriangle)
        plot!(p3, sigma_vec, BARKER_max, label = nothing, color = :blue)
        scatter!(p3, sigma_vec, infMALA_max, label = "infMALA", color = :purple, marker = :dtriangle)
        plot!(p3, sigma_vec, infMALA_max, label = nothing, color = :purple)
        scatter!(p3, sigma_vec, infmMALA_max, label = "infmMALA", color = :orange, marker = :xcross)
        plot!(p3, sigma_vec, infmMALA_max, label = nothing, color = :orange)
        title!(p3, "max ESJD of coordinates 2 to last")
        display(p3)
        savefig(p3, joinpath(data_save_directory, "Darcy_MCMC_max_ESJD_" * string(n_params) * "_" * string(output_dim) * ".png"))

        # ylim = (exp(-10), maximum([infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total])+0.1)
        ylim = (exp(-15), maximum([infmMALA_total; infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total]) * 1.1)
        gr(size = (1000, 600))
        p4 = plot(
            sigma_vec,
            MALA_min,
            xscale = :log10,
            yscale = :log10,
            label = "MALA",
            color = :red,
            marker = :diamond,
            xlabel = "proposal step-size",
            ylabel = "min ESJD",
            ylim = ylim,
            legend = :bottomright,
        )
        plot!(p4, sigma_vec, MALA_min, label = nothing, color = :red)
        scatter!(p4, sigma_vec, RW_min, label = "RW", color = :black, marker = :circle)
        plot!(p4, sigma_vec, RW_min, label = nothing, color = :black)
        scatter!(p4, sigma_vec, pCN_min, label = "pCN", color = :green, marker = :cross)
        plot!(p4, sigma_vec, pCN_min, label = nothing, color = :green)
        scatter!(p4, sigma_vec, BARKER_min, label = "Barker", color = :blue, marker = :utriangle)
        plot!(p4, sigma_vec, BARKER_min, label = nothing, color = :blue)
        scatter!(p4, sigma_vec, infMALA_min, label = "infMALA", color = :purple, marker = :dtriangle)
        plot!(p4, sigma_vec, infMALA_min, label = nothing, color = :purple)
        scatter!(p4, sigma_vec, infmMALA_min, label = "infmMALA", color = :orange, marker = :xcross)
        plot!(p4, sigma_vec, infmMALA_min, label = nothing, color = :orange)
        title!(p4, "min ESJD of coordinates 2 to last")
        display(p4)
        savefig(p4, joinpath(data_save_directory, "Darcy_MCMC_min_ESJD_" * string(n_params) * "_" * string(output_dim) * ".png"))

        # ylim = (exp(-10), maximum([infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total])+0.1)
       ylim = (exp(-15), maximum([infmMALA_total; infMALA_total; MALA_total; pCN_total; RW_total; BARKER_total]) * 1.1)
       gr(size = (1000, 600))
       p5 = plot(
           sigma_vec,
           MALA_total,
           xscale = :log10,
           yscale = :log10,
           label = "MALA",
           color = :red,
           marker = :diamond,
           xlabel = "proposal step-size",
           ylabel = "total ESJD",
           ylim = ylim,
           legend = :bottomright,
       )
       plot!(p5, sigma_vec, MALA_total, label = nothing, color = :red)
       scatter!(p5, sigma_vec, RW_total, label = "RW", color = :black, marker = :circle)
       plot!(p5, sigma_vec, RW_total, label = nothing, color = :black)
       scatter!(p5, sigma_vec, pCN_total, label = "pCN", color = :green, marker = :cross)
       plot!(p5, sigma_vec, pCN_total, label = nothing, color = :green)
       scatter!(p5, sigma_vec, BARKER_total, label = "Barker", color = :blue, marker = :utriangle)
       plot!(p5, sigma_vec, BARKER_total, label = nothing, color = :blue)
       scatter!(p5, sigma_vec, infMALA_total, label = "infMALA", color = :purple, marker = :dtriangle)
       plot!(p5, sigma_vec, infMALA_total, label = nothing, color = :purple)
       scatter!(p5, sigma_vec, infmMALA_total, label = "infmMALA", color = :orange, marker = :xcross)
       plot!(p5, sigma_vec, infmMALA_total, label = nothing, color = :orange)
       title!(p5, "total ESJD of coordinates 2 to last")
       display(p5)
       savefig(p5, joinpath(data_save_directory, "Darcy_MCMC_total_ESJD_" * string(n_params) * "_" * string(output_dim) * ".png"))

        =#









        #####################################################################
        # convergence analysis and others e.g. ESS, AP, second per iteration#
        #####################################################################
        d = n_params
        T = 10
        plots_id = 50:T
        num_repeats = 1
        methods = [MALASampling(), RWMHSampling(), pCNMHSampling(), BarkerSampling(), infMALASampling(), infmMALASampling()]
        #methods = [MALASampling(), RWMHSampling(), pCNMHSampling(), BarkerSampling(), infMALASampling()]
        # methods = [MALASampling(), RWMHSampling(), pCNMHSampling(), BarkerSampling()]

        # sigmas = [0.03, 0.071, 0.004, 0.125, 0.00246,]
        colors = [:red, :black, :green, :blue, :purple, :orange]


        samples_all = zeros(Float64, T, d*length(methods))
        d_t_values_all = zeros(Float64, T, length(methods))
        frob_values_all = zeros(Float64, T, length(methods))
        mse_values_all = zeros(Float64, T, length(methods))

        for (j, method) in enumerate(methods)
            sum_samples = zeros(Float64, T, d)
            sum_time = 0
            println(method)
            mcmc = MCMCWrapper(method, truth_sample, prior, emulator; init_params = u0)

            if method in [RWMHSampling(), pCNMHSampling()]
                new_step = optimize_stepsize(mcmc; init_stepsize = 0.01, N = 2000, discard_initial = 0)
            elseif method in [MALASampling(), BarkerSampling(), infMALASampling(), infmMALASampling()]
                new_step = optimize_stepsize_grad(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)
            elseif method in [infmMALASampling()]
                new_step = optimize_stepsize_grad(mcmc; init_stepsize = 0.15, N = 2000, discard_initial = 0)
            end# 0.1: 0.398ap 0.2: 0.63->0.04

            for i in 1:num_repeats
                start_time = time()
                chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = new_step, discard_initial = 0)
                end_time = time()
                sum_time += end_time - start_time
                #sum_time += (end_time - start_time).value

                acc_prob = accept_ratio(chain)
                println("acceptance probability: ", acc_prob)
                # sum_samples .+= chain.value[:, :, 1]
                posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

                posterior_samples = reduce(vcat, [get_distribution(posterior)[name] for name in get_name(posterior)])
                constrained_posterior = (mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1))'
                # Back to constrained coordinates
                sum_samples[:, :] += constrained_posterior
            end
            samples = sum_samples / num_repeats
            samples_all[:, d*(j-1)+1:d*j] = samples
            #println(samples)

            total_time = sum_time / num_repeats
            println("Total MCMC sampling time: ", total_time, " seconds")
            spiter = total_time / T
            println("spiter: ", spiter)

            ess_values = zeros(d)
            for i in 1:d
                sample_col = samples[:, i]
                autocorrelations = autocor(sample_col)
                ess_values[i] = T / (1 + 2 * sum(autocorrelations[2:end]))
            end

            println("Effective Sample Size (ESS): ", ess_values)
            println("Min ESS: ", minimum(ess_values))
            println("Median ESS: ", median(ess_values))
            println("Max ESS: ", maximum(ess_values))

            println("minESS/s: ", minimum(ess_values)/total_time)


            means = zeros(Float64, d)
            covariances = zeros(Float64, d, d, T)

            #= The true covariance matrix and means are picked from the 1 million runs of RWMH
post_mean
[-0.7847453632133576; 1.1536380127383883; -0.7527032367915321; -0.3065174841527967; -0.4302899954047373; 0.01676513587339504; -0.4471894270278742; -0.21494464278276107; -0.057366902334641916; 0.2133912881493806;;]
post_cov

D util
1.8026492844214957e11
            =#
            # (10 * 10, )
            Σ_true = [0.013543894717912987 -0.006214987713182169 0.010276097268447562 0.004919843945775869 0.0411939685184281 0.02893134140850419 0.00016198374158209465 0.0035932729791240303 -0.028155825864118314 0.0788705224122804;
            -0.006214987713182169 0.02626710857107154 0.00020699568534061525 0.02440185898745267 -0.018401616816766497 -0.012606503842269827 -0.004859127114068504 -0.017353445820157617 -0.03930058894315824 -0.04148768503564818;
            0.010276097268447562 0.00020699568534061525 0.023501193588531748 0.014996072185546048 0.026353402459610265 0.019441690983357976 -0.034975008632196956 -0.0032911503833491958 -0.05599423375680277 0.044247620799169295;
            0.004919843945775869 0.02440185898745267 0.014996072185546048 0.09536943851714826 0.021958055558362773 0.0002771168922271948 -0.03584587560423848 0.006876780722744529 -0.19908760669724637 -0.01475789852260678;
            0.0411939685184281 -0.018401616816766497 0.026353402459610265 0.021958055558362773 0.20285458253328026 0.09531025866016098 0.0254267962563691 0.014310186229978235 -0.10218624020842913 0.26490876237048616;
            0.02893134140850419 -0.012606503842269827 0.019441690983357976 0.0002771168922271948 0.09531025866016098 0.1573513923697528 -0.04830352045857495 -0.024721049716006076 0.004139961850327118 0.2051669969060487;
            0.00016198374158209465 -0.004859127114068504 -0.034975008632196956 -0.03584587560423848 0.0254267962563691 -0.04830352045857495 0.2615992388082379 0.007538685951571292 0.09455215696774284 0.07680598410365294;
            0.0035932729791240303 -0.017353445820157617 -0.0032911503833491958 0.006876780722744529 0.014310186229978235 -0.024721049716006076 0.007538685951571292 0.3403543280061304 -0.12182841179193125 -0.0025347701652472247;
            -0.028155825864118314 -0.03930058894315824 -0.05599423375680277 -0.19908760669724637 -0.10218624020842913 0.004139961850327118 0.09455215696774284 -0.12182841179193125 0.869562196231078 0.03202656568342419;
            0.0788705224122804 -0.04148768503564818 0.044247620799169295 -0.01475789852260678 0.26490876237048616 0.2051669969060487 0.07680598410365294 -0.0025347701652472247 0.03202656568342419 0.7216191324392598]

            mean_true = [-0.7847453632133576; 1.1536380127383883; -0.7527032367915321; -0.3065174841527967; -0.4302899954047373; 0.01676513587339504; -0.4471894270278742; -0.21494464278276107; -0.057366902334641916; 0.2133912881493806;;]

            # 10 * 10
            mean_true = [-1.0344906200007125; -1.031359704800955; 1.0166362686565669; 0.850028016116386; 1.4333275871098436; 1.0350609304625584; 0.7241519275532763; -1.1124012299325094; -1.1436817621907; -1.318690300902426;;]
            Σ_true = [0.0028424220854535976 -0.00039502042555249325 -0.0001941719958338421 0.00030455336970399826 -0.0027032635541074843 -0.0029338296089777196 -0.003593931242434306 0.0016278415078886954 -0.0030365281117883318 -0.007934014437569867; -0.00039502042555249325 0.005267802297403332 -0.001626695941963201 -0.005534831084914154 0.0007917937674129341 -0.0007277312362402547 -0.0013885934428028456 0.004952383956483919 0.000942360544461989 0.0024840261488643495; -0.0001941719958338421 -0.001626695941963201 0.0052178311131443 0.003400221834327995 -0.00040899327416662887 0.0002470474246207828 0.00593797599677156 0.00025837617541716424 -0.004116325873721924 -0.006106198936696026; 0.00030455336970399826 -0.005534831084914154 0.003400221834327995 0.016696711123912536 4.674316836050057e-5 -0.002309862814399651 0.0005395789256188748 -0.014244838388975316 0.0006451681589551268 -0.00934413571115772; -0.0027032635541074843 0.0007917937674129341 -0.00040899327416662887 4.674316836050057e-5 0.015608380045029346 0.0004485411429730017 -0.003693701272786808 -0.0029959370120195977 -0.0005425875700497006 0.0011838591688218768; -0.0029338296089777196 -0.0007277312362402547 0.0002470474246207828 -0.002309862814399651 0.0004485411429730017 0.020761729510300985 0.010322377679375154 0.008129735151636893 0.002179906476998882 0.005073365520603526; -0.003593931242434306 -0.0013885934428028456 0.00593797599677156 0.0005395789256188748 -0.003693701272786808 0.010322377679375154 0.034790339589332475 0.008383627541384055 0.004048185379492675 0.01180753948209379; 0.0016278415078886954 0.004952383956483919 0.00025837617541716424 -0.014244838388975316 -0.0029959370120195977 0.008129735151636893 0.008383627541384055 0.06319001416277097 -0.03608723897838521 0.018297611223755738; -0.0030365281117883318 0.000942360544461989 -0.004116325873721924 0.0006451681589551268 -0.0005425875700497006 0.002179906476998882 0.004048185379492675 -0.03608723897838521 0.0675530937052864 0.0013556628299626293; -0.007934014437569867 0.0024840261488643495 -0.006106198936696026 -0.00934413571115772 0.0011838591688218768 0.005073365520603526 0.01180753948209379 0.018297611223755738 0.0013556628299626293 0.07927712975243068]

            # 12 * 9
            mean_true =
            [1.0372023772893457; -0.722546776658123; 1.1656354638447883; 0.6390901251558959; -0.43340676731195105; -0.9189219913524974; 0.5009868097183949; 0.9383586267156909; 0.14834232507091655; -0.021553605779347926; 0.426096423434428; -0.3809507556861998;;]
            Σ_true = [0.011597522348189468 0.0021867155919923207 0.0020002901894734695 -0.0011164437482496068 -0.009971622161983773 -0.0013552002814056557 -0.0005137912559433265 -0.0026185566125702503 -0.007817222363624704 0.01236105464236234 0.041915447837749545 0.007467499693278485; 0.0021867155919923207 0.00805747893839138 -0.0008839174940166334 8.202851530533666e-5 -0.00018286237017708443 -0.0016144023369998146 0.009227773905137352 -0.00682108357553285 -0.013804499134617223 0.0022568049592148302 0.0069890345878497375 -0.0034840431079950647; 0.0020002901894734695 -0.0008839174940166334 0.012295285880988391 -0.007745234745069589 0.0008789529759478117 -0.0023126660344702054 -0.0072897076841610935 -0.01500380067885461 0.02602369053873404 0.0048382601187836865 0.0038394558230005534 -0.014920121872025783; -0.0011164437482496068 8.202851530533666e-5 -0.007745234745069589 0.055165596395557306 -0.015005187563373084 0.012617055312664537 -0.013730779185423063 0.02429174833529193 -0.022960412373230755 -0.04415761822389054 0.014239646240075747 0.07572749027354121; -0.009971622161983773 -0.00018286237017708443 0.0008789529759478117 -0.015005187563373084 0.08479780311523114 -0.010166905565297198 0.00926456964167927 0.009043945648259739 0.051651655794596005 0.05419221746303087 -0.07896641817137111 -0.012287843747527602; -0.0013552002814056557 -0.0016144023369998146 -0.0023126660344702054 0.012617055312664537 -0.010166905565297198 0.04304164218519579 0.0002957382204250789 0.017335932577272378 0.008161363855685139 -0.0640682734856499 -0.0023844164511593715 0.000784551143148301; -0.0005137912559433265 0.009227773905137352 -0.0072897076841610935 -0.013730779185423063 0.00926456964167927 0.0002957382204250789 0.0877234840683229 -0.011991922066710724 0.00630836381276619 -0.013101497541526928 -0.013962433047803684 -0.014760636560983175; -0.0026185566125702503 -0.00682108357553285 -0.01500380067885461 0.02429174833529193 0.009043945648259739 0.017335932577272378 -0.011991922066710724 0.1251135808685652 0.002174706556473975 -0.021128471307736516 -0.00578464519488972 0.04537036062122507; -0.007817222363624704 -0.013804499134617223 0.02602369053873404 -0.022960412373230755 0.051651655794596005 0.008161363855685139 0.00630836381276619 0.002174706556473975 0.33660207794115893 -0.03977622233586341 -0.06000960662438842 -0.02624607247290679; 0.01236105464236234 0.0022568049592148302 0.0048382601187836865 -0.04415761822389054 0.05419221746303087 -0.0640682734856499 -0.013101497541526928 -0.021128471307736516 -0.03977622233586341 0.33290581060525365 0.0035862856363109434 0.02713366757447015; 0.041915447837749545 0.0069890345878497375 0.0038394558230005534 0.014239646240075747 -0.07896641817137111 -0.0023844164511593715 -0.013962433047803684 -0.00578464519488972 -0.06000960662438842 0.0035862856363109434 0.19666584616664318 0.04210997326643107; 0.007467499693278485 -0.0034840431079950647 -0.014920121872025783 0.07572749027354121 -0.012287843747527602 0.000784551143148301 -0.014760636560983175 0.04537036062122507 -0.02624607247290679 0.02713366757447015 0.04210997326643107 0.24078723182412193]


            # 12 * 25
            mean_true =
            [0.8287269411835556; -1.0104114447317576; 0.962771715236735; 1.1373848474030377; 0.39330832972701074; -0.6638887094408431; -0.5210555138034721; -0.8381385276479716; -0.4297416001467203; 0.6979222395159808; 0.05313200148480844; -1.2250332657658196;;]
            Σ_true =
            [0.00526533229475111 -0.003684941354146558 0.0002535902907604644 -0.0038364280555043843 0.005589845189690968 -0.005588729573879874 0.0018836659416499443 0.006036297406207502 -0.007735685287131103 0.004552369190041187 0.022519039031899987 -0.010046592407850585; -0.003684941354146558 0.012632984870943692 -0.004687143036345746 0.0022895557654545604 -0.006772954143248092 0.006681349545796157 0.020687425403984085 0.0038583985650310405 0.0002471549690308552 -0.0059322492314922435 -0.020365390092731566 0.004335304511094494; 0.0002535902907604644 -0.004687143036345746 0.009198527207465947 -0.004614550784678668 -0.0002539244834629321 -0.0007868279836452292 -0.015613463068216966 -0.01312615682007049 -0.003844617449278286 0.004594815835801035 0.001869196512910324 -0.0007792165861419878; -0.0038364280555043843 0.0022895557654545604 -0.004614550784678668 0.03531036045597891 0.0010231150171398546 0.0011880112758059155 -0.021747156833011084 0.0008555357476025122 0.012340593401272254 -0.00449187112609315 -0.011102241226535638 0.030681907263428773; 0.005589845189690968 -0.006772954143248092 -0.0002539244834629321 0.0010231150171398546 0.05205098973377319 -0.010760918378246472 -0.015350570103768528 0.018524388896494076 0.015486233677571305 0.0035659891535095765 0.05714766535625897 -0.003982621725394278; -0.005588729573879874 0.006681349545796157 -0.0007868279836452292 0.0011880112758059155 -0.010760918378246472 0.030542904990741093 -0.01303317833542626 -0.009999244112751143 0.029776234504218357 0.008979537663330652 -0.018047696233329823 0.002248278125050541; 0.0018836659416499443 0.020687425403984085 -0.015613463068216966 -0.021747156833011084 -0.015350570103768528 -0.01303317833542626 0.14522254143681992 0.0457205407092449 -0.0652979448548847 -0.021101597815828592 -0.023407919362786984 -0.006570930947028777; 0.006036297406207502 0.0038583985650310405 -0.01312615682007049 0.0008555357476025122 0.018524388896494076 -0.009999244112751143 0.0457205407092449 0.07010113356485567 -0.010453087499132877 -0.006705860281023463 0.046865545915050884 -0.0074153898668723305; -0.007735685287131103 0.0002471549690308552 -0.003844617449278286 0.012340593401272254 0.015486233677571305 0.029776234504218357 -0.0652979448548847 -0.010453087499132877 0.11243427041052041 0.019722696955483766 0.020873248043362477 0.0064653288929429915; 0.004552369190041187 -0.0059322492314922435 0.004594815835801035 -0.00449187112609315 0.0035659891535095765 0.008979537663330652 -0.021101597815828592 -0.006705860281023463 0.019722696955483766 0.08227527287504884 0.03224149656689487 -0.03533242538517025; 0.022519039031899987 -0.020365390092731566 0.001869196512910324 -0.011102241226535638 0.05714766535625897 -0.018047696233329823 -0.023407919362786984 0.046865545915050884 0.020873248043362477 0.03224149656689487 0.1756924638103155 -0.02356376087802943; -0.010046592407850585 0.004335304511094494 -0.0007792165861419878 0.030681907263428773 -0.003982621725394278 0.002248278125050541 -0.006570930947028777 -0.0074153898668723305 0.0064653288929429915 -0.03533242538517025 -0.02356376087802943 0.1247031458146511]

            # 12 * 16
            mean_true =
            [0.8063899924134379; 0.8586236504597159; 0.8483578090315052; -0.712168521619151; -0.3394112115774971; -0.9138125286912948; 0.3455444191634136; 1.170562913808275; -0.2384183531622376; 0.07782200240069864; -0.5593133360109174; -0.03349116365558558;;]
            Σ_true =
            [0.016674480075106513 -0.00646527844499087 0.006095076478270636 -0.011921457309699093 -0.01284130064246639 -0.011230898989428448 0.026835279635121886 0.014650423045271001 0.03135233549425266 -0.013672572645531273 -0.036162447269003445 0.028967871011472444; -0.00646527844499087 0.022838750569539126 0.003691126897853387 0.010317093944869259 0.008394279747122257 -0.0030075314106071684 -0.031676809948237586 -0.021503423774953203 0.007086777443459076 -0.005311668804408066 0.02740358188889791 0.006148102662422218; 0.006095076478270636 0.003691126897853387 0.020094985540295627 -0.01703954252468702 -0.00037661393014900635 -0.008947581968984566 -0.013137077238081909 0.0029190436065868685 0.04076160482351068 -0.03115043184198103 0.004576634183049917 0.04926951295084687; -0.011921457309699093 0.010317093944869259 -0.01703954252468702 0.06662371917303443 0.0043406951659595945 0.015384534729076282 -0.008777474520743938 0.009267798494854282 -0.034284940466148266 0.036187460199159104 0.013075666659920058 -0.11085437115587783; -0.01284130064246639 0.008394279747122257 -0.00037661393014900635 0.0043406951659595945 0.06263710317280398 -0.008784433215081112 -0.029342351596549274 -0.030281688787949866 -0.017047813837166925 0.020236346760302424 0.06254260672787405 0.014219996575383396; -0.011230898989428448 -0.0030075314106071684 -0.008947581968984566 0.015384534729076282 -0.008784433215081112 0.061497392661565765 -0.035818752782170865 0.010720817899524757 -0.03174761133566813 0.0067114072160115975 -0.011660874126429306 -0.06797262219587827; 0.026835279635121886 -0.031676809948237586 -0.013137077238081909 -0.008777474520743938 -0.029342351596549274 -0.035818752782170865 0.14479831897543263 0.04097607854141921 0.007077716048212693 0.02116111300336263 -0.07650408455622458 -0.00724983780659022; 0.014650423045271001 -0.021503423774953203 0.0029190436065868685 0.009267798494854282 -0.030281688787949866 0.010720817899524757 0.04097607854141921 0.1552968392330557 0.02264225354149516 0.01424076927429715 -0.05706873369094948 -0.06735445000697235; 0.03135233549425266 0.007086777443459076 0.04076160482351068 -0.034284940466148266 -0.017047813837166925 -0.03174761133566813 0.007077716048212693 0.02264225354149516 0.16594601031034595 -0.07354942520707271 0.00836080993077616 0.15478487963047144; -0.013672572645531273 -0.005311668804408066 -0.03115043184198103 0.036187460199159104 0.020236346760302424 0.0067114072160115975 0.02116111300336263 0.01424076927429715 -0.07354942520707271 0.10484612965688427 0.00416529053620987 -0.07132809691637686; -0.036162447269003445 0.02740358188889791 0.004576634183049917 0.013075666659920058 0.06254260672787405 -0.011660874126429306 -0.07650408455622458 -0.05706873369094948 0.00836080993077616 0.00416529053620987 0.18641349090461462 0.06985902088847715; 0.028967871011472444 0.006148102662422218 0.04926951295084687 -0.11085437115587783 0.014219996575383396 -0.06797262219587827 -0.00724983780659022 -0.06735445000697235 0.15478487963047144 -0.07132809691637686 0.06985902088847715 0.4570309495202111]

            # 10 * 25
            if method == MALASampling()
                mean_true = [-1.003527804428099; 0.9655952382326473; 1.0482584266483508; 0.7889265305174673; -0.8240646039369008; -0.6985332501010314; 1.0114362313537137; 0.6583622687114513; 0.6541251830433298; 0.9093560348825408;;]
                Σ_true = [0.0008877466379108105 -0.0005900219359775587 -0.0003409203874732508 -0.0010653747329707533 5.4114785678974466e-5 0.0005642790211934296 -0.0007592332805988176 -0.0013035005575170414 -0.002426982339076719 -0.001290133669458191; -0.0005900219359775587 0.0019029068005339122 0.0003960425748348471 0.0019949473016054515 -0.0016668682008395223 -0.000817260069478739 0.0005485424604249323 0.002492071487299357 0.001116134985970365 0.0004279387614463592; -0.0003409203874732508 0.0003960425748348471 0.001208480833840667 0.0006594917678631758 -0.0006071277063671401 -0.0007912972383378277 0.0005067259476800462 9.080857624345024e-5 -0.000693170346278441 0.0006065794961142832; -0.0010653747329707533 0.0019949473016054515 0.0006594917678631758 0.0070927307836473824 -0.0013871795946037294 -0.0011307659154570515 0.0013214493300778784 0.002713990332897665 0.0019882542708440404 -0.00039432459040296517; 5.4114785678974466e-5 -0.0016668682008395223 -0.0006071277063671401 -0.0013871795946037294 0.009940856853857549 -0.0019112099489419004 0.0022604870999710014 -0.0031488739711853907 0.004052224282987638 -0.0028079996386031255; 0.0005642790211934296 -0.000817260069478739 -0.0007912972383378277 -0.0011307659154570515 -0.0019112099489419004 0.007571762406610121 -0.0037620061599963296 -0.0058199458148642256 0.0006349459042908787 0.0015558273578245077; -0.0007592332805988176 0.0005485424604249323 0.0005067259476800462 0.0013214493300778784 0.0022604870999710014 -0.0037620061599963296 0.011673295685023986 0.0030482288845495634 -0.0019547161763111003 -0.0020496551605430095; -0.0013035005575170414 0.002492071487299357 9.080857624345024e-5 0.002713990332897665 -0.0031488739711853907 -0.0058199458148642256 0.0030482288845495634 0.023656353933346214 -0.0030928382689313686 -0.0012833299441096196; -0.002426982339076719 0.001116134985970365 -0.000693170346278441 0.0019882542708440404 0.004052224282987638 0.0006349459042908787 -0.0019547161763111003 -0.0030928382689313686 0.02734871624054959 0.004326924246482567; -0.001290133669458191 0.0004279387614463592 0.0006065794961142832 -0.00039432459040296517 -0.0028079996386031255 0.0015558273578245077 -0.0020496551605430095 -0.0012833299441096196 0.004326924246482567 0.015175282629881381]
            elseif method == RWMHSampling()
                mean_true =[-1.0161889822439154; 0.9793847615319292; 1.0494469800236372; 0.8051365942695274; -0.8080911554605725; -0.7316793869699592; 1.0345521985741615; 0.7441421348252456; 0.7052054311627434; 0.9059792503232673;;]
                Σ_true = [0.002468790457049408 -0.0019657366625641065 -0.0009812133122522446 -0.0030784923665477465 0.0007566023921972829 0.0015938058807355969 -0.002312659642401504 -0.00527438828799129 -0.008391536881347882 -0.004952731940600256; -0.0019657366625641065 0.005804582815675443 0.0010034653782158645 0.0056951063079873025 -0.006144245847724386 -0.001906507819008034 0.0010687758208166595 0.00900791171473055 0.005065175958615292 0.0033156154646083917; -0.0009812133122522446 0.0010034653782158645 0.0034057413051833763 0.0017711614403685932 -0.0017179531507071767 -0.0019891047703706243 0.0011393194215589169 -0.0003174421129982354 -0.0021718640048300204 0.0017786060800969348; -0.0030784923665477465 0.0056951063079873025 0.0017711614403685932 0.0199372271785861 -0.00533221756523617 -0.0016227463016065674 0.0023084143853373644 0.006454249542036722 0.007281207790582748 0.0012457384855005806; 0.0007566023921972829 -0.006144245847724386 -0.0017179531507071767 -0.00533221756523617 0.032782519296100446 -0.006791130740578357 0.009316740371280918 -0.013638974710474596 0.00904188294528447 -0.014301324822037104; 0.0015938058807355969 -0.001906507819008034 -0.0019891047703706243 -0.0016227463016065674 -0.006791130740578357 0.021943275527757267 -0.012710390482083105 -0.01662181092178708 0.0003645612247961052 0.004871830996503184; -0.002312659642401504 0.0010687758208166595 0.0011393194215589169 0.0023084143853373644 0.009316740371280918 -0.012710390482083105 0.037997176354412604 0.008019216263452233 -0.00426800677895266 -0.006952598788054218; -0.00527438828799129 0.00900791171473055 -0.0003174421129982354 0.006454249542036722 -0.013638974710474596 -0.01662181092178708 0.008019216263452233 0.0790876619432671 0.0006822552863703999 0.005144038589632437; -0.008391536881347882 0.005065175958615292 -0.0021718640048300204 0.007281207790582748 0.00904188294528447 0.0003645612247961052 -0.00426800677895266 0.0006822552863703999 0.08559795570000964 0.016444459613144172; -0.004952731940600256 0.0033156154646083917 0.0017786060800969348 0.0012457384855005806 -0.014301324822037104 0.004871830996503184 -0.006952598788054218 0.005144038589632437 0.016444459613144172 0.05201534394085213]
            elseif method == pCNMHSampling()
                mean_true = [-0.9829295433337054; 0.9329163347658426; 1.0351289049677301; 0.733117161685246; -0.7788497143615091; -0.6456302061760687; 0.9407696204655726; 0.550793247647348; 0.621195290776481; 0.8826822442706612;;]
                Σ_true = [0.0022324626868160404 -0.0019389533763674546 -0.0011721990595168895 -0.003094205293249237 0.0007021615775576425 0.0014693649964007368 -0.0018882699177681843 -0.0034336699976181416 -0.006861951803600214 -0.004236252159430007; -0.0019389533763674546 0.005986864369948986 0.0013341528603170191 0.006314346416795451 -0.004870841885305364 -0.0028007123116600393 0.0015498657667557141 0.0069461671390366195 0.0050675553974249285 0.0013892851676541571; -0.0011721990595168895 0.0013341528603170191 0.00334184756992194 0.0023310308941264304 -0.0017074207517023333 -0.002954858270302099 0.0017931827160910473 0.000811686735460686 -0.00165393184168944 0.0022217078048779367; -0.003094205293249237 0.006314346416795451 0.0023310308941264304 0.021558190940805577 -0.004926620484066422 -0.0030486147759870544 0.002833503115410262 0.005242900505493457 0.007107441692082464 -0.0018455574522886348; 0.0007021615775576425 -0.004870841885305364 -0.0017074207517023333 -0.004926620484066422 0.027199598239116583 -0.005578207299998893 0.007784059767305402 -0.006003218198858346 0.004071335987102487 -0.010251190572806218; 0.0014693649964007368 -0.0028007123116600393 -0.002954858270302099 -0.0030486147759870544 -0.005578207299998893 0.0254849423673367 -0.012594396048804416 -0.020304328455509536 0.006211905046578114 0.00565359113515333; -0.0018882699177681843 0.0015498657667557141 0.0017931827160910473 0.002833503115410262 0.007784059767305402 -0.012594396048804416 0.033041636614314314 0.01038294169332423 -0.008972147314867028 -0.00791449834702399; -0.0034336699976181416 0.0069461671390366195 0.000811686735460686 0.005242900505493457 -0.006003218198858346 -0.020304328455509536 0.01038294169332423 0.0675821071434927 -0.009362925310106428 -0.0045505154073554; -0.006861951803600214 0.0050675553974249285 -0.00165393184168944 0.007107441692082464 0.004071335987102487 0.006211905046578114 -0.008972147314867028 -0.009362925310106428 0.07703358830820094 0.012510209016077214; -0.004236252159430007 0.0013892851676541571 0.0022217078048779367 -0.0018455574522886348 -0.010251190572806218 0.00565359113515333 -0.00791449834702399 -0.0045505154073554 0.012510209016077214 0.05397638435376395]
            elseif method == BarkerSampling()
                mean_true = [-1.0023204110972124; 0.9656307504969238; 1.0474382894915528; 0.7903812509255566; -0.8313268315897377; -0.6908846345899032; 1.0099778738061829; 0.6499502385989748; 0.6498548753357142; 0.9114965065893926;;]
                Σ_true = [0.0006132925729426496 -0.0004266298762834174 -0.00036384633227064765 -0.000780794053308504 0.00012669424510531306 0.0003876603232394473 -0.0005776542195492411 -0.0008580479994679366 -0.0012418222211603186 -0.0007653125816170454; -0.0004266298762834174 0.0013983480290322957 0.00033309137786560923 0.0012206969530573774 -0.000857311970870533 -0.0004939509223439868 0.00046013462061738806 0.001166651538476333 0.0008470234971871305 0.00042603113716845146; -0.00036384633227064765 0.00033309137786560923 0.0011065268910725285 0.0005664687390104707 -0.00037024840105067547 -0.0005360545027042337 0.0003745470956881464 0.00011459780625142302 -8.619838376634993e-5 0.0004739933744992727; -0.000780794053308504 0.0012206969530573774 0.0005664687390104707 0.0037799570349314683 -0.0009511574043043696 -0.0006596575691851257 0.0009349639706718883 0.00157147668200899 0.0015550670233712529 0.0005703822606993996; 0.00012669424510531306 -0.000857311970870533 -0.00037024840105067547 -0.0009511574043043696 0.004039367846351145 -0.0005960574966396975 0.0007965331375281552 -0.0008498751992908016 0.0009886394935301733 -0.0010307050282710494; 0.0003876603232394473 -0.0004939509223439868 -0.0005360545027042337 -0.0006596575691851257 -0.0005960574966396975 0.0034912393405112066 -0.001683605839817597 -0.002253160826196479 -0.00032005217191525013 0.000288445084634064; -0.0005776542195492411 0.00046013462061738806 0.0003745470956881464 0.0009349639706718883 0.0007965331375281552 -0.001683605839817597 0.0045290354367572205 0.0018271103767250154 0.0003259960218358691 -0.00018897631831400473; -0.0008580479994679366 0.001166651538476333 0.00011459780625142302 0.00157147668200899 -0.0008498751992908016 -0.002253160826196479 0.0018271103767250154 0.00846458219457758 0.001164929764209366 0.0002870968197166018; -0.0012418222211603186 0.0008470234971871305 -8.619838376634993e-5 0.0015550670233712529 0.0009886394935301733 -0.00032005217191525013 0.0003259960218358691 0.001164929764209366 0.009897530698738912 0.002134840831179369; -0.0007653125816170454 0.00042603113716845146 0.0004739933744992727 0.0005703822606993996 -0.0010307050282710494 0.000288445084634064 -0.00018897631831400473 0.0002870968197166018 0.002134840831179369 0.005702930867572982]
            elseif method == infMALASampling()
                mean_true = [-0.991630814842892; 0.9506837875441585; 1.040768439563442; 0.7640313277027077; -0.8030114364770873; -0.6760875769690896; 0.9893199572555982; 0.6060695225597316; 0.6310194481975642; 0.8780268461302182;;]
                Σ_true = [0.001191331262089195 -0.0008328599490455041 -0.0005391592480175598 -0.0015080587256771313 4.782861452449307e-5 0.0008195599187813241 -0.0013678531816955926 -0.0020007366803339527 -0.003388203348260117 -0.0018953919472439512; -0.0008328599490455041 0.0026679326654748085 0.0005306999151664794 0.002866171647107862 -0.0021557829664059546 -0.0012357941325566114 0.0010485865946906865 0.0034602286604020357 0.0016563370379730387 0.00013239090758902192; -0.0005391592480175598 0.0005306999151664794 0.0016928777387465463 0.0009678690408607823 -0.0005794037912148525 -0.0012740790132924576 0.0008971604739564797 2.531756274607381e-5 -0.0007195015784859433 0.0006670660269249451; -0.0015080587256771313 0.002866171647107862 0.0009678690408607823 0.01046605990494553 -0.0019522307629907185 -0.0014196531759872438 0.002424255506202772 0.003346937247266169 0.0026340235458373784 -0.0013700668056587989; 4.782861452449307e-5 -0.0021557829664059546 -0.0005794037912148525 -0.0019522307629907185 0.012606921038068793 -0.002199977539394512 0.002982951445425598 -0.004392182619532351 0.004214318752218476 -0.0026303326245471735; 0.0008195599187813241 -0.0012357941325566114 -0.0012740790132924576 -0.0014196531759872438 -0.002199977539394512 0.010621425506912733 -0.005384808222693012 -0.007756380151232007 0.000586967528430748 0.002048416439668175; -0.0013678531816955926 0.0010485865946906865 0.0008971604739564797 0.002424255506202772 0.002982951445425598 -0.005384808222693012 0.01693735525906599 0.004132744479757008 -0.0013159508620009504 -0.002293204858150974; -0.0020007366803339527 0.0034602286604020357 2.531756274607381e-5 0.003346937247266169 -0.004392182619532351 -0.007756380151232007 0.004132744479757008 0.03362596951820583 -0.0017909926194285405 -0.0017767601835779114; -0.003388203348260117 0.0016563370379730387 -0.0007195015784859433 0.0026340235458373784 0.004214318752218476 0.000586967528430748 -0.0013159508620009504 -0.0017909926194285405 0.0344629672447428 0.007270678108259427; -0.0018953919472439512 0.00013239090758902192 0.0006670660269249451 -0.0013700668056587989 -0.0026303326245471735 0.002048416439668175 -0.002293204858150974 -0.0017767601835779114 0.007270678108259427 0.02224203315612573]
            elseif method == infmMALASampling()
                mean_true = [-0.9881315296015112; 0.9481282464884795; 1.0389812130776188; 0.7647104777772259; -0.8151491761759545; -0.6707038725066284; 0.9916469757057264; 0.6015784825450907; 0.6150606471205489; 0.8704589272070166;;]
                Σ_true = [0.0010945877984134052 -0.0008365264035440502 -0.0005693760817948257 -0.0015490042418987182 0.00018964727687718135 0.0008488025680795242 -0.001203707185610524 -0.002006223970944267 -0.003076072834084443 -0.001813843213965599; -0.0008365264035440502 0.002550018391278771 0.0005926493744744856 0.002795713507379111 -0.002493399129229886 -0.0009938940418489153 0.0007581698120535561 0.0031252382370207514 0.0014488067068084312 0.0005033726359332611; -0.0005693760817948257 0.0005926493744744856 0.0016475441339459244 0.0010050771140683345 -0.0007942104966609731 -0.0011535463452176744 0.0008403745477889276 8.514531979673627e-6 -0.0005997133607419295 0.000904978726917798; -0.0015490042418987182 0.002795713507379111 0.0010050771140683345 0.009986336449006502 -0.00236077006570225 -0.0012097686721995716 0.0015932090534033626 0.0032316313196081832 0.0027412096391139225 -0.00015767015282108966; 0.00018964727687718135 -0.002493399129229886 -0.0007942104966609731 -0.00236077006570225 0.01378869247025668 -0.0028129930362264274 0.0034686847451722846 -0.0037471110705053243 0.004296660530031449 -0.004073027318377986; 0.0008488025680795242 -0.0009938940418489153 -0.0011535463452176744 -0.0012097686721995716 -0.0028129930362264274 0.010745045273459899 -0.005592465199089161 -0.008113477511616944 0.00022150739672314464 0.0020953152664425533; -0.001203707185610524 0.0007581698120535561 0.0008403745477889276 0.0015932090534033626 0.0034686847451722846 -0.005592465199089161 0.016573851927333426 0.004541869643351256 -0.0016813692881133943 -0.002850821723670535; -0.002006223970944267 0.0031252382370207514 8.514531979673627e-6 0.0032316313196081832 -0.0037471110705053243 -0.008113477511616944 0.004541869643351256 0.03243220873482862 -0.0018369770091692717 -0.0016989532388317096; -0.003076072834084443 0.0014488067068084312 -0.0005997133607419295 0.0027412096391139225 0.004296660530031449 0.00022150739672314464 -0.0016813692881133943 -0.0018369770091692717 0.03181746268094255 0.005076644298906218; -0.001813843213965599 0.0005033726359332611 0.000904978726917798 -0.00015767015282108966 -0.004073027318377986 0.0020953152664425533 -0.002850821723670535 -0.0016989532388317096 0.005076644298906218 0.021650765540474175]
            end

            for t in plots_id
                sample_subset = samples[1:t, :]
                #=
                if t == 1
                    println("Skipping t=1 because it cannot be computed with a sample.")
                    continue
                end
                =#
                covariances[:, :, t] = cov(sample_subset, dims=1)
                #if t%5000 == 0
                #    println(t, cov(sample_subset, dims=1))
                #end
            end

            for t in plots_id # 2:T
                Σ_t = covariances[:, :, t]
                d_t_values_all[t, j] = 1/sqrt(n_params) * sqrt( sum((log.(diag(Σ_t)) .- log.(diag(Σ_true))).^2) )
                #d_t_values_all[t, j] = sqrt(mean((diag(Σ_t) .- diag(Σ_true))).^2)
            end
        #=
            for t in plots_id
                if t == 1
                    println("Skipping t=1 because it cannot be computed with a sample.")
                    continue
                end
                Σ_t = covariances[:, :, t]
                d_cov = size(Σ_t, 1)
                d_t_values_all[t, j] = 1 / sqrt(d_cov) * sqrt( sum((log.(diag(Σ_t)) .- log.(diag(Σ_true))).^2) )
                println("aa", d_t_values_all[t, j])
            end
            d_t_values_all[:,j] = cumsum(d_t_values_all[:,j]) ./ collect(1:length(plots_id))
        =#

            for t in plots_id
                Σ_t = covariances[:, :, t]
                frob_values_all[t, j] = norm(Σ_t - Σ_true)
            end

            for t in plots_id
                sample_mean = mean(samples[1:t, :], dims=1)
                mse_values_all[t, j] = mean((sample_mean[:] - mean_true).^2)
                #if t%5000 == 0
                #    println(sample_mean - mean_true')
                #end
            end
        end
        gr(size = (1000, 600))
        # traceplots
        p_tp = plot(1:T, samples_all[:, 1], label="MALA",
        color=colors[1], xlabel="Iteration (t)", ylabel="sample", title="trace plots for 1st coordinate", lw=2)
        plot!(1:T, samples_all[:, 11].+1.0, label="RW", color=colors[2], lw=2)
        plot!(1:T, samples_all[:, 21].+2.0, label="pCN", color=colors[3], lw=2)
        plot!(1:T, samples_all[:, 31].+3.0, label="BARKER", color=colors[4], lw=2)
        plot!(1:T, samples_all[:, 41].+4.0, label="infMALA", color=colors[5], lw=2)
        plot!(1:T, samples_all[:, 51].+5.0, label="infmMALA", color=colors[6], lw=2)
        display(p_tp)
        savefig(p_tp, joinpath(data_save_directory, "Darcy_MCMC_traceplots_1st_" * string(n_params) * "_" * string(output_dim) * ".png"))

        p_tp = plot(1:T, samples_all[:, 2], label="MALA",
        color=colors[1], xlabel="Iteration (t)", ylabel="sample", title="trace plots for 2nd coordinate", lw=2)
        plot!(1:T, samples_all[:, 12].+1.0, label="RW", color=colors[2], lw=2)
        plot!(1:T, samples_all[:, 22].+2.0, label="pCN", color=colors[3], lw=2)
        plot!(1:T, samples_all[:, 32].+3.0, label="BARKER", color=colors[4], lw=2)
        plot!(1:T, samples_all[:, 42].+4.0, label="infMALA", color=colors[5], lw=2)
        plot!(1:T, samples_all[:, 52].+5.0, label="infmMALA", color=colors[6], lw=2)
        display(p_tp)
        savefig(p_tp, joinpath(data_save_directory, "Darcy_MCMC_traceplots_2nd_" * string(n_params) * "_" * string(output_dim) * ".png"))


        plot_d_t = plot(plots_id, d_t_values_all[plots_id, 1], label="d_t MALA",
        color=colors[1], xlabel="Iteration (t)", ylabel="d_t",
        title="Convergence of d_t over iterations", lw=2)
        plot!(plots_id, d_t_values_all[plots_id, 2], label="d_t RW", color=colors[2], lw=2)
        plot!(plots_id, d_t_values_all[plots_id, 3], label="d_t pCN", color=colors[3], lw=2)
        plot!(plots_id, d_t_values_all[plots_id, 4], label="d_t BARKER", color=colors[4], lw=2)
        plot!(plots_id, d_t_values_all[plots_id, 5], label="d_t infMALA", color=colors[5], lw=2)
        plot!(plots_id, d_t_values_all[plots_id, 6], label="d_t infmMALA", color=colors[6], lw=2)
        display(plot_d_t)
        savefig(plot_d_t, joinpath(data_save_directory, "Darcy_MCMC_cov_conv_" * string(n_params) * "_" * string(output_dim) * ".png"))

        plot_frob_norm = plot(plots_id, frob_values_all[plots_id, 1],
        label="Frobenius MALA", color=colors[1], xlabel="Iteration (t)", ylabel="Frobenius Norm",
        title="Convergence of Frobenius Norm over iterations", lw=2)
        plot!(plots_id, frob_values_all[plots_id, 2], label="Frobenius RW", color=colors[2], lw=2)
        plot!(plots_id, frob_values_all[plots_id, 3], label="Frobenius pCN", color=colors[3], lw=2)
        plot!(plots_id, frob_values_all[plots_id, 4], label="Frobenius BARKER", color=colors[4], lw=2)
        plot!(plots_id, frob_values_all[plots_id, 5], label="Frobenius infMALA", color=colors[5], lw=2)
        plot!(plots_id, frob_values_all[plots_id, 6], label="Frobenius infmMALA", color=colors[6], lw=2)
        display(plot_frob_norm)
        savefig(plot_frob_norm, joinpath(data_save_directory, "Darcy_MCMC_frob_conv_" * string(n_params) * "_" * string(output_dim) * ".png"))

        plot_mse = plot(plots_id, mse_values_all[plots_id, 1], label="MSE MALA",
        color=colors[1], xlabel="Iteration (t)", ylabel="MSE",
        title="Convergence of MSE over iterations", lw=2)
        plot!(plots_id, mse_values_all[plots_id, 2], label="MSE RW", color=colors[2], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 3], label="MSE pCN", color=colors[3], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 4], label="MSE BARKER", color=colors[4], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 5], label="MSE infMALA", color=colors[5], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 6], label="MSE infmMALA", color=colors[6], lw=2)
        display(plot_mse)
        savefig(plot_mse, joinpath(data_save_directory, "Darcy_MCMC_mse_conv_" * string(n_params) * "_" * string(output_dim) * ".png"))

        plot_mse = plot(plots_id, mse_values_all[plots_id, 1],
        xscale = :log10, yscale = :log10, label="MSE MALA",
        color=colors[1], xlabel="Iteration (t)", ylabel="MSE",
        title="(logscaled) Convergence of MSE over iterations", lw=2)
        plot!(plots_id, mse_values_all[plots_id, 2], label="MSE RW", color=colors[2], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 3], label="MSE pCN", color=colors[3], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 4], label="MSE BARKER", color=colors[4], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 5], label="MSE infMALA", color=colors[5], lw=2)
        plot!(plots_id, mse_values_all[plots_id, 6], label="MSE infmMALA", color=colors[6], lw=2)
        display(plot_mse)
        savefig(plot_mse, joinpath(data_save_directory, "Darcy_MCMC_logscale_mse_conv_" * string(n_params) * "_" * string(output_dim) * ".png"))











        #=
        gr(size = (1500, 400), legend = false)
        p1 = contour(
            pts_per_dim,
            pts_per_dim,
            κ_post_mean',    # analogue to κ_true', κ_ens_mean'
            fill = true,
            levels = 15,
            title = "kappa posterior mean",
            colorbar = true
        )

        p2 = contour(
            pts_per_dim,
            pts_per_dim,
            κ_post_ptw_var',
            fill = true,
            levels = 15,
            title = "kappa posterior var",
            colorbar = true,
        )

        post_h_2d = solve_Darcy_2D(darcy, κ_post_mean)
        p3 = contour(pts_per_dim, pts_per_dim, post_h_2d', fill = true, levels = 15, title = "pressure", colorbar = true)
        l = @layout [a b c]
        plt = plot(p1, p2, p3, layout = l)
        savefig(plt, joinpath(fig_save_directory, "output_post_GP.png")) # pre update
        # or savefig(plt, joinpath(fig_save_directory, "output_it_" * string(n_iter) * ".png")) # pre update
        # println("Final coefficients (ensemble mean):")
        # println(get_u_mean_final(ekiobj))    # or println(get_u_mean_final(ekiobj))



        function compute_ess(samples::Matrix{Float64})
            n, d = size(samples)
            ess_values = zeros(d)

            for i in 1:d
                autocov = autocor(sample_col[i])
                ess_values[i] = n / (1 + 2 * sum(autocov[2:end]))
            end

            return ess_values
        end




       p_tp = plot(1:T, posterior_samples[:, 1:2], label="MALA",
       color=colors[1], xlabel="Iteration (t)", ylabel="d_t", title="samples", lw=2)
       figpath = joinpath(figure_save_directory, "posterior_MCMC_" * case * ".pdf")
       savefig(figpath)
       figpath = joinpath(figure_save_directory, "posterior_MCMC_" * case * ".png")
       savefig(figpath)
        =#










        # Save data
        save(
            joinpath(data_save_directory, "posterior.jld2"),
            "posterior",
            posterior,
            "input_output_pairs",
            input_output_pairs,
            "truth_params",
            truth_params,
        )
    end
end

main()
