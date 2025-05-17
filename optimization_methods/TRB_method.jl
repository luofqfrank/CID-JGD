using LinearAlgebra
using JuMP 
using Ipopt
using DelimitedFiles

# trust region bundle method
# params = Dict("tol" => 1e-6, "omega" => 1.2, "max_bundle_size" => 100, "gamma" => 0.01, "gamma_bar" => 0.1, "gamma_hat" => 0.4, "max_T" => 320)
function TRB_method(f::Function, f_cpt::Function, x_init, params, file_path)
    if file_path != nothing
        file = open(file_path, "w")
        println(file, "iter    obj    delta    eval_time    QP_time    iter_time")
    end
    tol = params["tol"]
    max_bundle_size = params["max_bundle_size"]
    gamma = params["gamma"]
    gamma_bar = params["gamma_bar"]
    gamma_hat = params["gamma_hat"]
    omega = params["omega"]
    x = x_init
    n = length(x_init)
    obj = f(n, x)[2]
    R_bar = 0.1 # trust-region radius
    Q = zeros(n,n)
    k = 0
    delta = 1
    max_dist = 1
    start_time = time() 
    elapsed_time = 0
    while elapsed_time <= params["max_T"]
        iter_time = time()
        l = 1
        y = x
        eval_time = time()
        c, obj = f(n, y)
        g = autoGrad(f_cpt, n, c, y)
        obj_x = f(n, x)[2]
        eval_time = time() - eval_time
        e = 0.0
        b = 0.0
        B_oracle = [(e, b, g, y)]
        R = min(1, R_bar) 
        rho = 0
        QP_time = 0
        while rho < gamma 
            # solving subproblem
            T1 = time()
            y = subproblem_solver(x, obj_x, Q, B_oracle, R, omega)
            T2 = time() 
            QP_time += T2 - T1
            T1 = time()
            code_y, obj_y = f(n, y)
            delta = obj_x - phi(obj_x, x, y, B_oracle, omega)
            T2 = time()
            eval_time += T2 - T1 
            # compute rho 
            rho = (obj_x - obj_y) / (obj_x - Phi(obj_x, x, y, Q, B_oracle, omega))
            # update working model
            B_oracle_prev = [tup for tup in B_oracle]
            if length(B_oracle) >= max_bundle_size && length(B_oracle) >= 3 
                popfirst!(B_oracle)
            end
            T1 = time()
            g = autoGrad(f_cpt, n, code_y, y)
            T2 = time() 
            eval_time += T2 - T1 
            b, e = compute_b_e(f, omega, x, y, g)
            push!(B_oracle, (e, b, g, y))
            #push!(B_agg, (alpha_agg, g_agg))
            # compute the second test rho_bar
            rho_bar = (obj_x - phi(obj_x, x, y, B_oracle, omega)) / (obj_x - Phi(obj_x, x, y, Q, B_oracle_prev, omega))
            R = rho_bar < gamma_bar ? R : max(0.5 * R, 1e-6)
            l += 1
            elapsed_time = time() - start_time
            if elapsed_time > params["max_T"]
                break
            end
        end 
        x = y 
        R_bar = rho < gamma_hat ? R : 2 * R
        k += 1 
        elapsed_time = time() - start_time
        iter_time = time() - iter_time
        @printf("iter = %d, obj = %f, delta = %f, eval_time = %f, QP_time = %f, iter_time = %f\n", k, obj, delta, eval_time, QP_time, iter_time)
        if file_path != nothing 
            println(file, "$k    $obj    $delta    $eval_time    $QP_time    $iter_time")
        end
        if elapsed_time > params["max_T"]
            break
        end
        if delta <= params["tol"] # termination criterion
            break 
        end
    end
    if file_path != nothing 
        close(file)
    end
end

# solving the mibimization problem min Phi at Step 2
# B_oracle: list of elements in the form (e, b, g, y)
# B_agg: list of elements in the form (alpha, g)
function subproblem_solver(x_in, obj_in, Q, B_oracle, R, omega)
    n = length(x_in)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_optimizer_attributes(model, "tol" => 1e-16,  # Desired optimality tolerance
        "constr_viol_tol" => 1e-16,  # Desired constraint violation tolerance
        "max_iter" => 10000)
    @variable(model, y[1:n])
    @variable(model, w)
    @variable(model, u)
    @objective(model, Min, w + 0.5 * (y - x_in)' * Q * (y - x_in))
    @constraint(model, w >= obj_in + u)
    for (e, b, g, z) in B_oracle
        alpha = compute_alpha(omega, b, e)
        @constraint(model, u >= - alpha + dot(g + omega * (y - x_in), y - x_in))
    end
    @constraint(model, (y - x_in)' * (y - x_in) <= R^2)
    # Solve the optimization problem
    optimize!(model)
    # retrieve the optimal solution and optimal objective
    y_opt = value.(y)
    return y_opt
end

# g is the gradient at y
function compute_b_e(f::Function, omega, x, y, g)
    n = length(x)
    b = 0.5 * norm(y - x)^2
    e = f(n, x)[2] - f(n, y)[2] - dot(g, x - y)
    return b, e  
end

function compute_alpha(omega, b, e)
    return max(e, 0) + omega * b 
end

function phi(obj_in, x_in, y, B_oracle, omega)
    res = obj_in 
    ls = []
    for (e, b, g, z) in B_oracle 
        alpha = compute_alpha(omega, b, e)
        push!(ls, - alpha + dot(g + omega * (y - x_in), y - x_in))
    end
    res += maximum(ls)
    return res 
end

function Phi(obj_in, x_in, y, Q, B_oracle, omega)
    res = 0.5 * (y - x_in)' * Q * (y - x_in)
    res += phi(obj_in, x_in, y, B_oracle, omega)
    return res 
end