using LinearAlgebra
using JuMP 
using Ipopt
using DelimitedFiles

# reference: global convergent variable metric method for nonconvex nondifferentiable unconstrained minimization by J. Vlcek and L. Luksan
# Algorithm 2.1 in the reference
function variable_metric_method(f::Function, f_cpt::Function, x_init, params, file_path)
    file = open(file_path, "w")
    println(file, "Iter     CPU_time     obj_dec     obj     flag_opt     flag_obj     flag_timing")
    epsilon = params["epsilon"] # final accuracy tolerance
    t_aux = params["t_aux"] # auxiliary stepsize
    t_max = params["t_max"] # stepsize bound
    rho = params["rho"] # correction parameter 
    L = params["L"] # correction parameter
    C = params["C"] # matrix scaling parameter
    D = params["D"] # direction vector length control
    n = length(x_init)
    # initialization
    x = x_init
    H_hat = I(n)
    y = x 
    alpha = 0
    code, obj = f(n, x)
    g = autoGrad(f_cpt, n, code, x)
    mu = 1
    i_C = i_E = i_S = i_U = 0
    n_C = n_S = 0
    k = 0 # iteration counter
    g_t = g # tilde g 
    alpha_t = 0 
    w = 1
    t_I = 0
    t_L = 0
    t_R = 1
    m = k # m is the latest iteration achieving serious step
    g_m = g 
    x_code_list = []
    local_grad_norm = 1
    start_time = time() 
    elapsed_time = 0
    flag_opt = flag_obj = flag_timing = false 
    obj_prev = Inf 
    obj_cur = obj 
    obj_dec_list = [Inf]
    println(file, "0     0     0     $obj     $flag_opt     $flag_obj     $flag_timing")
    while !flag_opt && !flag_obj && !flag_timing 
        # correction
        Hg = H_hat * g_t 
        if norm(Hg) > D 
            H_hat = H_hat * D / norm(Hg) 
        end
        w_hat = g_t' * H_hat * g_t + 2 * alpha_t 
        if w_hat < rho * norm(g_t)^2 || i_C == i_U == 1 
            w = w_hat + rho * norm(g_t)^2
            H = H_hat + rho * I(n)
        else
            w = w_hat 
            H = H_hat 
        end
        if n_C >= L 
            i_C = 1
        end
        # line search
        d = - H * g_t 
        n_S = n_S + 1
        if i_E == 0
            t_I = 0.5 * (t_aux + t_max)
        else
            t_I = 2 * t_L
            i_E = 0 
        end
        t_L, t_R, alpha, beta = line_search(f, f_cpt, x, d, t_I, t_aux, w, D)
        x = x + t_L * d 
        y = y + t_R * d 
        code1, obj1 = f(n, x)
        code2, obj2 = f(n, y)
        if isinf(obj1) || isinf(obj2)
            break 
        end
        push!(x_code_list, (y, code2))
        if t_L > 0 
            obj_prev = obj_cur 
            obj_cur = obj1 
            push!(x_code_list, (x, code1))
            pushfirst!(obj_dec_list, obj_prev - obj_cur)
        end
        g = autoGrad(f_cpt, n, code2, y)
        # update preparation
        u = g - g_m 
        g_m = t_L > 0 ? g : g_m 
        mu = (2 * mu + min(C^2, 0.1)) / 3
        if t_L > 0 # descent step 
            n_S, i_S, H, mu = matrix_scaling(mu, i_S, n_S, C, H)
            H_hat, i_U = BFGS_update(H, u, d, t_L, t_max, rho)
            # descent step initialization
            k = k + 1
            g_t, g_m, alpha_t, m = descent_step_init(g, k)
        else 
            g_t_prev = g_t 
            g_t, alpha_t = aggregation(g_m, g, g_t, alpha, alpha_t, H)
            H_hat, i_U = SR1_update(H, u, d, g_t_prev, g_t, rho, i_C, t_R)
            k = k + 1
        end
        pts_local = get_points_in_range(f, x, 10 * epsilon, x_code_list)
        local_grad_norm = norm(joint_gradient_QP_solver([autoGrad(f_cpt, n, c, z) for (z, c) in pts_local], nothing)[1])
        flag_opt = local_grad_norm < 1e2 * epsilon ? true : false 
        flag_obj = (length(obj_dec_list) >= 10 && maximum(obj_dec_list[1:10]) <= 1e-8) ? true : false 
        flag_timing = elapsed_time >= 1200 ? true : false  
        end_time = time()
        elapsed_time = end_time - start_time 
        println(file, "$k     $elapsed_time     $(obj_dec_list[1])     $obj_cur     $flag_opt     $flag_obj     $flag_timing")
        println("k = ", k, ", obj = ", obj1)
    end 
    close(file)
end


function descent_step_init(g, k)
    g_t = g 
    g_m = g
    alpha_t = 0
    m = k
    return g_t, g_m, alpha_t, m 
end 


function matrix_scaling(mu, i_S, n_S, C, H)
    i_S = mu > 1 ? i_S + 1 : i_S 
    if mu > C && n_S > 3 && i_S > 1 
        n_S = 0
        i_S = 0
        H = mu * H 
        mu = sqrt(mu)
    end
    return n_S, i_S, H, mu
end

# g_t_k: tilde_g[k]
# g_t_kp1: tilde_g[k+1]
function SR1_update(H, u, d, g_t_k, g_t_kp1, rho, i_C, t_R)
    n = length(d)
    v = H * u - t_R * d 
    if dot(g_t_k, v) < 0 && (i_C == 0 || rho * norm(g_t_kp1)^2 <= dot(g_t_kp1, v)^2 / dot(u, v) && rho * n <= norm(v)^2 / dot(u, v))
        i_U = 1
        H_hat = H - v * v' / dot(u, v)
    else 
        i_U = 0 
        H_hat = H 
        flag = false
    end
    return H_hat, i_U
end

function BFGS_update(H, u, d, t_L, t_max, rho)
    if norm(u) == 0 && t_L <= 0.5 * t_max
        i_E = 1
    end
    if dot(u, d) > rho 
        i_U = 1 
        H_hat = H + (t_L + u' * H * u / dot(u, d)) * d * d' / dot(u, d) - (H * u * d' + d * u' * H) / dot(u, d)
    else 
        i_U = 0 
        H_hat = H 
    end
    return H_hat, i_U 
end

# g_m: g[m] 
# g: g[k+1]
# g_t: tilde_g[k]
# alpha: alpha[k+1]
# alpha_t: tilde_alpha[k] 
# H: H[k]
function aggregation(g_m, g, g_t, alpha, alpha_t, H)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_optimizer_attributes(model, "tol" => 1e-8,  # Desired optimality tolerance
        "constr_viol_tol" => 1e-8,  # Desired constraint violation tolerance
        "max_iter" => 10000)  # Maximum number of iterations
    @variable(model, lambda[1:3] >= 0)
    @constraint(model, sum(lambda[i] for i in 1:3) == 1)
    @objective(model, Min, (lambda[1] * g_m + lambda[2] * g + lambda[3] * g_t)' * H * (lambda[1] * g_m + lambda[2] * g + lambda[3] * g_t)
                            + 2 * (lambda[2] * alpha + lambda[3] * alpha_t))
    optimize!(model) 
    opt_lambda = value.(lambda)
    g_t = opt_lambda[1] * g_m + opt_lambda[2] * g + opt_lambda[3] * g_t
    alpha_t = opt_lambda[2] * alpha + opt_lambda[3] * alpha_t
    return g_t, alpha_t 
end

# reference: global convergent variable metric method for nonconvex nondifferentiable unconstrained minimization by J. Vlcek and L. Luksan
# page 8, Line Search Procedure
# omega >= 1: a locality measure parameter 
# D > 0: step length control parameter, it can be set as a maximum reasonable distance in one step.  
function line_search(f::Function, f_cpt::Function, x, d, t_I, t_aux, w, D)
    n = length(x)
    omega = 1.5
    gamma = 0.5
    c_A = 1 / 8
    c_L = 1 / 10
    c_R = 1 / 3
    c_T = 1 / 8
    kappa = 1 / 4
    flag = true
    t_A = 0
    t = t_U = t_I  
    _, obj_0 = f(n, x) 
    t_L, t_R = nothing, nothing
    while flag 
        println("t = ", t)
        code, obj = f(n, x + t * d)
        g = autoGrad(f_cpt, n, code, x + t * d)
        beta = max(abs(obj_0 - obj + t * dot(d, g)), gamma * (t * norm(d))^omega)
        if obj <= obj_0 - c_T * t * w 
            t_A = t 
            if t >= t_aux || beta > c_A * w 
                t_L = t_R = t 
                alpha = 0
                return t_L, t_R, alpha, beta 
            end
        else
            t_U = t 
        end
        if - beta + dot(d, g) >= - c_R * w && (t - t_A) * norm(d) <= D 
            t_L = 0
            t_R = t 
            alpha = beta 
            return t_L, t_R, alpha, beta
        end
        t = 0.5 * (t_A + t_U)
    end
end

