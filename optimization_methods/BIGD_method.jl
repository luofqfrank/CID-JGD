using LinearAlgebra
using JuMP 
using OSQP
using DelimitedFiles
using Printf

#=
params = Dict("epsilon0" => 1e-3, "nu0" => 1e-3, "epsilon_opt" => 1e-5, "nu_opt" => 1e-4, 
              "gamma" => 0.5, "theta_epsilon" => 0.1, "theta_nu" => 0.9,
              "t_lb" => 1e-8, "rho_0" => 1e-2, "max_T" => 320)
=#

function BIGD_method(f::Function, f_cpt::Function, x_init, params, access, file_path)
    if file_path != nothing
        file = open(file_path, "w")
        println(file, "iter    obj    grad    epsilon    eval_time    QP_time    iter_time    act_codes    tot_codes")
    end
    iter = 1
    x = x_init 
    epsilon = params["epsilon0"]
    nu = params["nu0"]
    n = length(x)
    code, obj = f(n, x)
    code2point = Dict(code => (x, obj, autoGrad(f_cpt, n, code, x)))
    start_time = time()
    elapsed_time = 0
    flag = 1
    while elapsed_time <= params["max_T"] && flag == 1
        iter_time = time() 
        # compute descent direction
        g, act_codes, eval_time, QP_time = gradient(f_cpt, code2point, epsilon, x, access)
        # check termination condition
        if norm(g) <= params["nu_opt"] && epsilon <= params["epsilon_opt"]
            obj = f(n, x)[2]
            flag = 0 
        end
        t = line_search(f, f_cpt, code2point, x, g, params["t_lb"], params["gamma"], params["rho_0"], epsilon, act_codes)
        if norm(g) <= nu
            nu = max(params["theta_nu"] * nu, params["nu_opt"]) 
            epsilon = max(params["theta_epsilon"] * epsilon, params["epsilon_opt"]) 
        end
        x = x - t * g / norm(g)
        code, obj = f(n, x) 
        code2point[code] = (x, obj, autoGrad(f_cpt, n, code, x))
        iter = iter + 1
        iter_time = time() - iter_time
        @printf("iter = %d, obj = %f, g = %f, epsilon = %f, eval_time = %f, QP_time = %f, iter_time = %f, act_codes = %d, tot_codes = %d\n", iter, obj, norm(g), epsilon, eval_time, QP_time, iter_time, length(act_codes), length(code2point))
        if file_path != nothing 
            println(file, "$iter    $obj    $(norm(g))    $epsilon    $eval_time    $QP_time    $iter_time    $(length(act_codes))    $(length(code2point))")
        end
        elapsed_time = time() - start_time
    end
    if file_path != nothing 
        close(file)
    end
    return x, obj
end

# gradient calculation within a radius
function gradient(f_cpt, code2point, radius, x, access)
    eval_time = 0
    QP_time = 0
    n = length(x)
    act_codes = Set()
    grad_list = []
    for code in keys(code2point) 
        (z, val, g) = code2point[code]
        if norm(x - z) <= radius
            push!(act_codes, code)
            T = time()
            if access == 1
                g = autoGrad(f_cpt, n, code, x) 
            end
            push!(grad_list, g)
            eval_time += time() - T 
        end
    end
    k = length(grad_list)
    @assert k >= 1
    QP_time = time()
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
    QP_time = time() - QP_time
    # retrieve the optimal solution and optimal objective
    opt_lambda = value.(lambda)
    opt_vec = sum(opt_lambda[i] * grad_list[i] for i in 1:k)
    return opt_vec, act_codes, eval_time, QP_time
end

function line_search(f::Function, f_cpt::Function, code2point, x, g, t_lb, gamma, rho, radius, act_codes)
    n = length(x)
    d = g / norm(g)
    t = 1
    obj = f(n, x)[2]
    flag = 0
    while t >= t_lb
        x_trial = x - t * d
        code_trial, obj_trial = f(n, x_trial)
        code2point_update(f_cpt, code2point, code_trial, x_trial, obj_trial, x)
        if obj - obj_trial >= rho * t * norm(g)
            flag = 1
            break
        else 
            t = gamma * t 
        end
    end
    return t
end

function code2point_update(f_cpt::Function, code2point, code, x, val, x0)
    n = length(x)
    if !haskey(code2point, code)
        code2point[code] = (x, val, autoGrad(f_cpt, n, code, x))
    elseif norm(x - x0) < norm(code2point[code][1] - x0)
        code2point[code] = (x, val, autoGrad(f_cpt, n, code, x))
    end
end

function autoGrad(f_cpt::Function, n, code, x_val) 
    df_dx = ForwardDiff.gradient(x -> f_cpt(n, code, x), x_val)
    return df_dx 
end