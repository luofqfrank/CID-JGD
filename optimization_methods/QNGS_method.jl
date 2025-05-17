using LinearAlgebra
using JuMP 
using Ipopt
using DelimitedFiles

#=
params = Dict("grad_opt" => 1e-4, "epsilon_opt" => 1e-5, "epsilon_0" => 0.1, "nu" => 1, "psi" => 0.5, "xi" => 1e-4, "eta_lb" => 1e-9, "eta_ub" => 0.9,
              "alpha_lb" => 1e-4, "alpha_ub" => 1, "gamma" => 0.5, "J_lb" => 5, "J_ub" => 10,
              "N0" => 100, "mu_lb" => 0.2, "mu_ub" => 100, "w_lb" => 1e-4, "w_ub" => 1, "m" => 100, "max_T" => 320)
=#
# quasi-newton gradient sampling method
function QNGS_method(f::Function, f_cpt::Function, x_init, params, file_path) 
    if file_path != nothing 
        file = open(file_path, "w")
        println(file, "iter    obj    epsilon    grad    eval_time    QP_time    iter_time    sample_size")
    end 
    start_time = time() 
    elapsed_time = 0
    n = length(x_init)
    x = x_init
    W = I(n)
    epsilon = params["epsilon_0"]
    X = [x]
    p = 0
    c, obj = f(n, x)
    g = autoGrad(f_cpt, n, c, x)
    G = reshape(g, n, 1)
    y = dual_QP_solver(G, W)
    k = 0 
    st_list = []
    x_prev = nothing
    g_prev = nothing
    local_grad_norm = 1
    Gy_norm = norm(G * y)
    while elapsed_time <= params["max_T"]
        elapsed_time = time() - start_time
        iter_time = time()
        QP_time = 0
        eval_time = 0
        T1 = time()
        y = dual_QP_solver(G, W)
        T2 = time()
        QP_time += T2 - T1
        Gy_norm = norm(G * y)
        d = - W * G * y 
        alpha = AW_line_search(f, f_cpt, x, G, W, d, y, p, params)
        x_prev = x 
        x = x + alpha * d
        T1 = time()
        c, obj = f(n, x) 
        g_prev = g
        g = autoGrad(f_cpt, n, c, x)
        T2 = time() 
        eval_time += T2 - T1 
        Gy_W = metric_norm(G*y, W)
        if Gy_W <= params["nu"] * epsilon && Gy_W >= params["xi"] * norm(d) && alpha > 0 
            epsilon = params["psi"] * epsilon 
        end
        T1 = time()
        X, p = sample_set_update(f, X, x, epsilon, G, W, d, y, alpha, params)
        sample_size = length(X)
        W = inv_Hessian_approx_update(alpha, x_prev, x, g_prev, g, d, y, W, G, st_list, params)
        G =  generate_G(f, f_cpt, X)
        T2 = time()
        eval_time += T2 - T1 
        k += 1
        elapsed_time = time() - start_time 
        iter_time = time() - iter_time
        @printf("iter = %d, obj = %f, epsilon = %f, grad = %f, eval_time = %f, QP_time = %f, iter_time = %f, sample_size = %d\n", k, obj, epsilon, Gy_norm, eval_time, QP_time, iter_time, sample_size)
        if file_path != nothing 
            println(file, "$k    $obj    $epsilon    $Gy_norm    $eval_time    $QP_time    $iter_time    $sample_size")
        end
        # termination check
        if epsilon <= params["epsilon_opt"] && Gy_norm <= params["grad_opt"] 
            break
        end
    end
    if file_path != nothing 
        close(file)
    end
end


function dual_QP_solver(G, W)
    n, m = size(G)
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

    @variable(model, y[1:m] >= 0)
    @objective(model, Min, (G * y)' * W * (G * y))
    @constraint(model, sum(y[i] for i = 1:m) == 1)
    optimize!(model)
    # retrieve the optimal solution and optimal objective
    y_opt = value.(y)
    obj_value = objective_value(model) 
    return y_opt 
end

function metric_norm(x, W)
    return sqrt(max(x' * W * x, 0))
end

function armijo_condition(f::Function, x, y, d, alpha, G, W, params)
    n = length(x)
    return f(n, x)[2] - f(n, x + alpha*d)[2] > params["eta_lb"] * alpha * metric_norm(G*y, W)^2
end

function curvature_condition(f::Function, f_cpt::Function, x, y, d, alpha, G, W, params) 
    n = length(x)
    c1 = f(n, x)[1]
    c2 = f(n, x + alpha * d)[1]
    g1 = autoGrad(f_cpt, n, c1, x)
    g2 = autoGrad(f_cpt, n, c2, x + alpha * d)
    return dot(g2, d) >= params["eta_ub"] * dot(g1, d)
end

function wolfe_condition(f::Function, f_cpt::Function, x, y, d, alpha, G, W, params)
    return armijo_condition(f, x, y, d, alpha, G, W, params) && curvature_condition(f, f_cpt, x, y, d, alpha, G, W, params) 
end

# return alpha
function AW_line_search(f::Function, f_cpt::Function, x, G, W, d, y, N, params)
    alpha_ub = params["alpha_ub"]
    gamma = params["gamma"]
    N0 = params["N0"]
    J_lb = params["J_lb"]
    J_ub = params["J_ub"]
    l = 0
    u = alpha_ub 
    j = 0
    alpha = params["gamma"] * params["alpha_ub"]
    if norm(d) < 1e-10
        return alpha 
    end
    while true  
        if N < N0 && j > J_ub 
            alpha = 0
            return alpha 
        end
        l = j > J_lb ? 0 : l 
        if wolfe_condition(f, f_cpt, x, y, d, alpha, G, W, params) || (armijo_condition(f, x, y, d, alpha, G, W, params) && j > J_lb)
            return alpha 
        end
        if !armijo_condition(f, x, y, d, alpha, G, W, params)
            u = alpha 
        else 
            l = alpha 
        end
        alpha = (1 - gamma) * l + gamma * u 
        j += 1
    end
end


function sampling(N, x_c, epsilon)
    n = length(x_c)
    sp_lst = []
    for i = 1:N 
        v = 2 * rand(n) .- 1
        v = epsilon * v * rand() / norm(v) 
        push!(sp_lst, x_c + v)
    end
    return sp_lst 
end

# X is a list of sample points
function sample_set_update(f::Function, X, x_in, epsilon, G, W, d, y, alpha, params)
    if metric_norm(G * y, W) >= params["xi"] * norm(d)^2 && alpha >= params["alpha_lb"]
        X = [x_in]
        N = 0
        return X, N 
    end
    X_new = []
    for x in X 
        if norm(x - x_in) <= epsilon 
            push!(X_new, x)
        end
    end
    push!(X_new, x_in)
    N_old = length(X_new) - 1
    K = 50
    X_bar = sampling(K, x_in, epsilon)
    n = length(x_in)
    X = vcat(X_new, X_bar)
    idx = min(max(length(X) - params["N0"], 0) + 1,  N_old + 1)
    X = X[idx:length(X)]
    N = length(X) - 1
    return X, N
end

# X: list of sample points
function generate_G(f::Function, f_cpt::Function, X)
    g_list = []
    for x in X 
        n = length(x)
        c = f(n, x)[1]
        g = autoGrad(f_cpt, n, c, x)
        push!(g_list, g)
    end
    G = hcat(g_list...)
    return G 
end

function compute_delta(s, t, W, params)
    delta = dot(s,t) >= params["mu_lb"] * t' * W * t ? 1 : (1 - params["mu_lb"]) * t' * W * t / (t' * W * t - dot(s,t))
    return delta 
end

# st_list: list of (s,t) computed in previous iterations
function inv_Hessian_approx_update(alpha, x1, x2, g1, g2, d, y, W, G, st_list, params)
    n = length(x1)
    s = x2 - x1 
    t = g2 - g1 
    push!(st_list, (s, t))
    if norm(s) < 1e-10 || norm(t) < 1e-10 
        W = W 
    elseif metric_norm(G * y, W) >= params["xi"] * norm(d)^2 && alpha >= params["alpha_lb"]
        delta = compute_delta(s, t, W, params)
        r = delta * s + (1 - delta) * W * t 
        if abs(dot(r,t)) > 1e-12
            W = (I(n) - r * t'/ dot(r,t)) * W * (I - t * r' / dot(r,t)) + r * r' / dot(r,t)
        end
    else 
        w = 0.5 * (params["w_lb"] + params["w_ub"])
        W = w * I(n)
        m = params["m"]
        for (s, t) in st_list[max(end-m+1,1):end]
            delta = compute_delta(s, t, W, params)
            r = delta * s + (1 - delta) * W * t
            if min(norm(s), norm(t)) > 1e-10 && abs(dot(r,t)) > 1e-12 && max(norm(r)^2, norm(t)^2) <= params["mu_ub"] * dot(r, t)
                W = (I(n) - r * t'/ dot(r,t)) * W * (I - t * r' / dot(r,t)) + r * r' / dot(r,t) 
            end
        end
    end
    if !is_psd(W)
        W = I(n)
    end
    return W 
end

function is_psd(A; tol=1e-8)
    # Check if the matrix is symmetric/Hermitian
    if !ishermitian(A)
        return false
    end
    # Compute eigenvalues
    eigvals_A = eigvals(Hermitian(A))  # Ensures real eigenvalues
    return all(eigvals_A .≥ tol)  # Allow small numerical errors
end