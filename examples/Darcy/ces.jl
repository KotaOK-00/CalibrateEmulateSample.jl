# # [Learning the Pearmibility field in a Darcy flow from noisy sparse observations]

# In this example we hope to illustrate function learning. One may wish to use function learning in cases where the underlying parameter of interest is actual a finite-dimensional approximation (e.g. spatial discretization) of some "true" function. Treating such an object directly will lead to increasingly high-dimensional learning problems as the spatial resolution is increased, resulting in poor computational scaling and increasingly ill-posed inverse problems. Treating the object as a discretized function from a function space, one can learn coefficients not in the standard basis, but instead in a basis of this function space, it is commonly the case that functions will have relatively low effective dimension, and will be depend only on the spatial discretization due to discretization error, that should vanish as resolution is increased.

# We will solve for an unknown permeability field ``\kappa`` governing the pressure field of a Darcy flow on a square 2D domain. To learn about the permeability we shall take few pointwise measurements of the solved pressure field within the domain. The forward solver is a simple finite difference scheme taken and modified from code [here](https://github.com/Zhengyu-Huang/InverseProblems.jl/blob/master/Fluid/Darcy-2D.jl).

# First we load standard packages
using LinearAlgebra
using Distributions
using Random
using JLD2

# the package to define the function distributions
import GaussianRandomFields # we wrap this so we don't want to use "using"
const GRF = GaussianRandomFields

# and finally the EKP packages
using CalibrateEmulateSample
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.ParameterDistributions
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses

# We include the forward solver here
include("GModel.jl")

# Then link some outputs for figures and plotting
fig_save_directory = joinpath(@__DIR__, "output")
data_save_directory = joinpath(@__DIR__, "output")
if !isdir(fig_save_directory)
    mkdir(fig_save_directory)
end
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end# TOML interface for fitting parameters of a sinusoid

PLOT_FLAG = true
if PLOT_FLAG
    using Plots
    @info "Plotting enabled, this will reduce code performance. Figures stored in $fig_save_directory"
end

# Set a random seed.
seed = 100234
rng = Random.MersenneTwister(seed)





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
    for i in [20, 16, 13, 11, 10]
        # Define the spatial domain and discretization
        dim = 2
        N, L = 80, 1.0
        pts_per_dim = LinRange(0, L, N)
        obs_ΔN = i # output observations: (N / obs_ΔN - 1)^2
        #20, 16, 13, 11, 10 of obs_ΔN correspond to

        # 9, 16, 25, 36, 49

        # To provide a simple test case, we assume that the true function parameter is a particular sample from the function space we set up to define our prior.
        # More precisely we choose a value of the truth that doesnt have a vanishingly small probability under the prior defined by a probability distribution over functions; here taken as a family of Gaussian Random Fields (GRF).
        # The function distribution is characterized by a covariance function - here a Matern kernel which assumes a level of smoothness over the samples from the distribution.
        # We define an appropriate expansion of this distribution, here based on the Karhunen-Loeve expansion (similar to an eigenvalue-eigenfunction expansion) that is truncated to a finite number of terms, known as the degrees of freedom (`dofs`). The `dofs` define the effective dimension of the learning problem, decoupled from the spatial discretization. Explicitly, larger `dofs` may be required to represent multiscale functions, but come at an increased dimension of the parameter space and therefore a typical increase in cost and difficulty of the learning problem.

        smoothness = 0.1
        corr_length = 1.0
        dofs = 12 # input dimensions

        grf = GRF.GaussianRandomField(
            GRF.CovarianceFunction(dim, GRF.Matern(smoothness, corr_length)),
            GRF.KarhunenLoeve(dofs),
            pts_per_dim,
            pts_per_dim,
        )
        eigenfuncs = grf.data.eigenfunc
        println("KL basis (eigenfunctions): ", size(eigenfuncs))

        # We define a wrapper around the GRF, and as the permeability field must be positive we introduce a domain constraint into the function distribution. Henceforth, the GRF is interfaced in the same manner as any other parameter distribution with regards to interface.
        pkg = GRFJL()
        distribution = GaussianRandomFieldInterface(grf, pkg) # our wrapper from EKP
        domain_constraint = bounded_below(0) # make κ positive
        pd = ParameterDistribution(
            Dict("distribution" => distribution, "name" => "kappa", "constraint" => domain_constraint),
        ) # the fully constrained parameter distribution

        # Now we have a function distribution, we sample a reasonably high-probability value from this distribution as a true value (here all degrees of freedom set with `u_{\mathrm{true}} = -0.5`). We use the EKP transform function to build the corresponding instance of the ``\kappa_{\mathrm{true}}``.
        # u_true = -1.5 * ones(dofs, 1)
        u_true = sign.(randn(dofs, 1)) # the truth parameter
        println("True coefficients: ")
        println(u_true)
        κ_true = transform_unconstrained_to_constrained(pd, u_true) # builds and constrains the function.
        println("κ_true: ", size(κ_true))
        κ_true = reshape(κ_true, N, N)
        println("κ_true reshaped: ", size(κ_true))
        # Now we generate the data sample for the truth in a perfect model setting by evaluating the the model here, and observing it by subsampling in each dimension every `obs_ΔN` points, and add some observational noise
        darcy = Setup_Param(pts_per_dim, obs_ΔN, κ_true)
        println(" Number of observation points: $(darcy.N_y)")
        h_2d_true = solve_Darcy_2D(darcy, κ_true)
        y_noiseless = compute_obs(darcy, h_2d_true)
        obs_noise_cov = 0.25^2 * I(length(y_noiseless)) * (maximum(y_noiseless) - minimum(y_noiseless))
        truth_sample = vec(y_noiseless + rand(rng, MvNormal(zeros(length(y_noiseless)), obs_noise_cov)))


        # Now we set up the Bayesian inversion algorithm. The prior we have already defined to construct our truth
        prior = pd


        # We define some algorithm parameters, here we take ensemble members larger than the dimension of the parameter space
        N_ens = 30 # number of ensemble members
        N_iter = 10 # number of EKI iterations

        # We sample the initial ensemble from the prior, and create the EKP object as an EKI algorithm using the `Inversion()` keyword
        initial_params = construct_initial_ensemble(rng, prior, N_ens)
        ekiobj = EKP.EnsembleKalmanProcess(initial_params, truth_sample, obs_noise_cov, Inversion())

        # We perform the inversion loop. Remember that within calls to `get_ϕ_final` the EKP transformations are applied, thus the ensemble that is returned will be the positively-bounded permeability field evaluated at all the discretization points.
        println("Begin inversion")
        err = []
        final_it = [N_iter]
        for i in 1:N_iter
            params_i = get_ϕ_final(prior, ekiobj)
            g_ens = run_G_ensemble(darcy, params_i)
            terminate = EKP.update_ensemble!(ekiobj, g_ens)
            push!(err, get_error(ekiobj)[end]) #mean((params_true - mean(params_i,dims=2)).^2)
            println("Iteration: " * string(i) * ", Error: " * string(err[i]))
            if !isnothing(terminate)
                final_it[1] = i - 1
                break
            end
        end
        n_iter = final_it[1]
        # We plot first the prior ensemble mean and pointwise variance of the permeability field, and also the pressure field solved with the ensemble mean. Each ensemble member is stored as a column and therefore for uses such as plotting one needs to reshape to the desired dimension.
        if PLOT_FLAG
            gr(size = (1500, 400), legend = false)
            prior_κ_ens = get_ϕ(prior, ekiobj, 1)
            println("size of prior_κ_ens: ", size(prior_κ_ens))
            println("size of mean of prior_κ_ens: ", size(mean(prior_κ_ens, dims = 2)))
            κ_ens_mean = reshape(mean(prior_κ_ens, dims = 2), N, N)
            p1 = contour(
                pts_per_dim,
                pts_per_dim,
                κ_ens_mean',
                fill = true,
                levels = 15,
                title = "kappa mean",
                colorbar = true,
            )
            κ_ens_ptw_var = reshape(var(prior_κ_ens, dims = 2), N, N)
            p2 = contour(
                pts_per_dim,
                pts_per_dim,
                κ_ens_ptw_var',
                fill = true,
                levels = 15,
                title = "kappa var",
                colorbar = true,
            )
            h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
            p3 = contour(pts_per_dim, pts_per_dim, h_2d', fill = true, levels = 15, title = "pressure", colorbar = true)
            l = @layout [a b c]
            plt = plot(p1, p2, p3, layout = l)
            savefig(plt, joinpath(fig_save_directory, "output_prior.png")) # pre update

        end

        # Now we plot the final ensemble mean and pointwise variance of the permeability field, and also the pressure field solved with the ensemble mean.
        if PLOT_FLAG
            gr(size = (1500, 400), legend = false)
            final_κ_ens = get_ϕ_final(prior, ekiobj) # the `ϕ` indicates that the `params_i` are in the constrained space
            println("size of final_κ_ens: ", size(final_κ_ens))
            κ_ens_mean = reshape(mean(final_κ_ens, dims = 2), N, N)
            println("size of κ_ens_mean: ", size(κ_ens_mean))
            p1 = contour(
                pts_per_dim,
                pts_per_dim,
                κ_ens_mean',
                fill = true,
                levels = 15,
                title = "kappa mean",
                colorbar = true,
            )
            κ_ens_ptw_var = reshape(var(final_κ_ens, dims = 2), N, N)
            println("size of _ptw_var: ", size(κ_ens_ptw_var))
            p2 = contour(
                pts_per_dim,
                pts_per_dim,
                κ_ens_ptw_var',
                fill = true,
                levels = 15,
                title = "kappa var",
                colorbar = true,
            )
            h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
            p3 = contour(pts_per_dim, pts_per_dim, h_2d', fill = true, levels = 15, title = "pressure", colorbar = true)
            l = @layout [a b c]
            plt = plot(p1, p2, p3; layout = l)
            savefig(plt, joinpath(fig_save_directory, "output_it_" * string(n_iter) * ".png")) # pre update

        end
        println("Final coefficients (ensemble mean):")
        println(get_u_mean_final(ekiobj))

        # We can compare this with the true permeability and pressure field:
        if PLOT_FLAG
            gr(size = (1000, 400), legend = false)
            p1 = contour(pts_per_dim, pts_per_dim, κ_true', fill = true, levels = 15, title = "kappa true", colorbar = true)
            p2 = contour(
                pts_per_dim,
                pts_per_dim,
                h_2d_true',
                fill = true,
                levels = 15,
                title = "pressure true",
                colorbar = true,
            )
            l = @layout [a b]
            plt = plot(p1, p2, layout = l)
            savefig(plt, joinpath(fig_save_directory, "output_true.png"))
        end

        # Finally the data is saved
        u_stored = get_u(ekiobj, return_array = false)
        g_stored = get_g(ekiobj, return_array = false)

        save(
            joinpath(data_save_directory, "calibrate_results.jld2"),
            "inputs",
            u_stored,
            "outputs",
            g_stored,
            "eigenfuncs",
            eigenfuncs,
            "pts_per_dim",
            pts_per_dim,
            "prior",
            prior,
            "eki",
            ekiobj,
            "darcy",
            darcy,
            "truth_sample",
            truth_sample, #data
            "truth_input_constrained", # the discrete true parameter field
            κ_true,
            "truth_input_unconstrained", # the discrete true KL coefficients
            u_true,
        )

        # emulate-smaple
        cases = [
            "GP", # diagonalize, train scalar GP, assume diag inputs
        ]

        #### CHOOSE YOUR CASE:
        mask = [1] # 1:8 # e.g. 1:8 or [7]
        for (case) in cases[mask]


            println("case: ", case)
            min_iter = 1
            max_iter = 10 # number of EKP iterations to use data from is at most this

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

            # First let's run a short chain to determine a good step size
            mcmc = MCMCWrapper(RWMHSampling(), truth_sample, prior, emulator; init_params = u0)
            new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

            #mcmc = MCMCWrapper(infMALASampling(), truth_sample, prior, emulator; init_params = u0)
            #new_step = optimize_stepsize_grad(mcmc; init_stepsize = 0.01, N = 2000, discard_initial = 0)

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

            posterior_samples = reduce(vcat, [get_distribution(posterior)[name] for name in get_name(posterior)])
            constrained_posterior = (mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1))'
            # Back to constrained coordinates
#auto

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
        if i == 20
            p5 = plot(0:lags, autocorrelation, label=label, lw=2, linestyle=linestyles[i])
        else
            plot!(p5, 0:lags, autocorrelation, label=label, lw=2, linestyle=linestyles[i])
            display(plot_mse)
        savefig(plot_mse, joinpath(data_save_directory, "Darcy_MCMC_logscale_mse_conv_" * string(n_params) * "_" * string(output_dim) * ".png"))
        end
    end
end

main()
