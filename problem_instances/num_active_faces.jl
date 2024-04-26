# encoding rule: n+2 tuple
# (idx, sgn1, sgn2, ..., sgn_{n+1})
# idx = 1, 2, ..., n+1; sgn_i = -1, 1
# idx = i --> g(x_i) for i in 1 ... n
# idx = n+1 --> g(-x1-x2-...-xn)

function num_active_faces(n, x)
    sgn = zeros(Int, n+1)
    vec = zeros(Float64, n+1)
    for i = 1:n 
        if x[i] >= 0
            sgn[i] = 1 
            vec[i] = log(x[i] + 1)
        else 
            sgn[i] = -1
            vec[i] = log(- x[i] + 1)
        end
    end
    if sum(-x[i] for i = 1:n) >= 0
        sgn[n+1] = 1
        vec[n+1] = log(sum(-x[i] for i = 1:n) + 1)
    else 
        sgn[n+1] = -1 
        vec[n+1] = log(sum(x[i] for i = 1:n) + 1)
    end 
    res = maximum(vec)
    idx = argmax(vec)
    code = vcat(idx, sgn) 
    code = tuple(code...)
    return code, res 
end

function num_active_faces_cpt(n, code, x)
    idx = code[1]
    sgn = zeros(Int, n+1)
    for i = 2:(n+2)
        sgn[i-1] = code[i]
    end
    if idx >= 1 && idx <= n
        res = log(sgn[idx] * x[idx] + 1)
    else
        res = log(sgn[n+1] * sum(-x[i] for i = 1:n) + 1)
    end
    return res 
end

function num_active_faces_init(n)
    x_init = zeros(Float64, n)
    for i = 1:n 
        x_init[i] = 1
    end
    return x_init 
end