using LinearAlgebra
using JuMP 
using Ipopt
using OSQP
using DelimitedFiles

#=
params = Dict("epsilon0" => 1e-3, "nu0" => 1e-3, "beta" => 1e-16, "gamma" => 0.5, "m" => 2 * n,
              "epsilon_opt" => 1e-5, "nu_opt" => 1e-4, "theta_epsilon" => 0.1, "theta_nu" => 0.9, "max_T" => 320)
=#
# gradient sampling method
function GS_method(f::Function, f_cpt::Function, x_init, params, file_path)
    if file_path != nothing
        file = open(file_path, "w")
        println(file, "iter    obj    grad    epsilon    eval_time    QP_time    iter_time    samples    tot_samples")
    end
    iter = 1
    x = x_init 
    epsilon = params["epsilon0"]
    nu = params["nu0"]
    n = length(x)
    obj = f(n,x)[2]
    start_time = time()
    elapsed_time = 0
    flag = 1
    while elapsed_time <= params["max_T"] && flag == 1
        iter_time = time()
        # sampling
        eval_time = time()
        sp_lst = sampling(params["m"], x, epsilon)
        eval_time = time() - eval_time
        # compute descent direction
        QP_time = time()
        g = QP_solver(f, f_cpt, sp_lst)[1]
        QP_time = time() - QP_time
        # check termination condition
        if norm(g) <= params["nu_opt"] && epsilon <= params["epsilon_opt"]
            flag = 0 
        end
        if norm(g) <= nu 
            nu = params["theta_nu"] * nu 
            epsilon = max(params["theta_epsilon"] * epsilon, params["epsilon_opt"]) 
            t = 0 
        else 
            t = line_search(f, x, g, params["beta"], params["gamma"])
        end
        x = x - t * g / norm(g)
        obj = f(n, x)[2] 
        iter = iter + 1
        iter_time = time() - iter_time 
        @printf("iter = %d, obj = %f, g = %f, epsilon = %f, eval_time = %f, QP_time = %f, iter_time = %f, samples = %d, tot_samples = %d\n", iter, obj, norm(g), epsilon, eval_time, QP_time, iter_time, params["m"], iter*params["m"])
        if file_path != nothing 
            println(file, "$iter    $obj    $(norm(g))    $epsilon    $eval_time    $QP_time    $iter_time    $(params["m"])    $(iter*params["m"])")
        end
        elapsed_time = time() - start_time
    end
    if file_path != nothing 
        close(file)
    end
    return x, obj 
end

function QP_solver(f::Function, f_cpt::Function, sp_list)
    k = length(sp_list)
    @assert k >= 1
    n = length(sp_list[1])
    grad_list = []
    for pt in sp_list
        code, val = f(n, pt)
        g = autoGrad(f_cpt, n, code, pt) 
        push!(grad_list, g)
    end    
    model = Model(OSQP.Optimizer)
    # Set OSQP parameters
    set_optimizer_attributes(model,
        "verbose" => false,     # Disable solver output
        "rho" => 0.1,          # ADMM penalty parameter (default: 1.0)
        "max_iter" => 10000,    # Increase iteration limit
        "eps_abs" => 1e-8,      # Absolute tolerance (default: 1e-3)
        "eps_rel" => 1e-8,      # Relative tolerance
        "warm_start" => true    # Enable warm-starting
    )

    @variable(model, lambda[1:k] >= 0)
    @constraint(model, sum(lambda[i] for i in 1:k) == 1)
    #@objective(model, Min, sum(v[i]^2 for i in 1:n))
    @objective(model, Min, sum(sum(lambda[j] * grad_list[j][i] for j in 1:k)^2 for i in 1:n))  
    # Solve the optimization problem
    optimize!(model) 
    # retrieve the optimal solution and optimal objective
    opt_lambda = value.(lambda)
    opt_vec = sum(opt_lambda[i] * grad_list[i] for i in 1:k)
    return opt_vec, opt_lambda
end

# sample N points from the cube of length 2 * epsilon centered at x
function sampling(N, x_c, epsilon)
    n = length(x_c)
    sp_lst = []
    push!(sp_lst, x_c)
    for i = 1:N 
        v = 2 * rand(n) .- 1
        v = epsilon * v * rand() / norm(v) 
        push!(sp_lst, x_c + v)
    end
    return sp_lst 
end

function line_search(f::Function, x, g, beta, gamma)
    flag = 0
    n = length(x)
    obj = f(n, x)[2]
    d = - g / norm(g)
    t = 1
    while flag == 0 && t > beta 
        if f(n, x + t*d)[2] < obj - beta * t * norm(g)
            flag = 1
        else 
            t = gamma * t 
        end
    end
    return t
end