# encoding rule: n-1 tuple
# (idx1, idx2, ..., idx_(n-1)): idx_i is 1 or 2 or 3

function Chained_CB3_I(n, x)
    vec = zeros(Float64, n-1, 3)
    code = zeros(Int, n-1)
    res = 0
    for i = 1:(n-1)
        vec[i, 1] = x[i]^4 + x[i+1]^2
        vec[i, 2] = (2 - x[i])^2 + (2 - x[i+1])^2
        vec[i, 3] = 2 * exp(- x[i] + x[i+1])
        res += maximum(vec[i, :])
        code[i] = argmax(vec[i, :])
    end
    code = tuple(code...)
    return code, res 
end

# the component function of Chained_CB3_I
function Chained_CB3_I_cpt(n, code, x)
    res = 0
    for i in 1:(n-1)
        if code[i] == 1
            res += x[i]^4 + x[i+1]^2
        elseif code[i] == 2
            res += (2 - x[i])^2 + (2 - x[i+1])^2
        else 
            res += 2 * exp(- x[i] + x[i+1])
        end
    end
    return res 
end

function Chained_CB3_I_init(n) 
    x_init = zeros(Float64, n)
    for i in 1:n 
        x_init[i] = 2 
    end
    return x_init
end