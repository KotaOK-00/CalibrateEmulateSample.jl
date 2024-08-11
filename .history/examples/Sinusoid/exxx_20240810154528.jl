using KernelFunctions

# Define the squared exponential kernel with ARDTransform
rbf_len = [1.9952706691900783, 3.066374123568536]  # Length scales for each dimension
rbf = KernelFunctions.SqExponentialKernel() ∘ ARDTransform(rbf_len)

# Define input data points
x1 = [0.0, 0.0]
x2 = [1.0, 1.0]

# Calculate the kernel value using ARDTransform
kval_julia = rbf(x1, x2)

# Manually calculate the kernel value
dist_sq = sum(((x1 .- x2) ./ rbf_len) .^ 2)
kval_manual = exp(-0.5 * dist_sq)

# Compare the results
println("Kernel value using ARDTransform: ", kval_julia)
println("Manually calculated kernel value: ", kval_manual)
println("Are the values equal? ", isapprox(kval_julia, kval_manual))
